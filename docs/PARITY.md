# Numerical parity with the PyTorch reference

The port is verified stage by stage against upstream ScaleLSD
(`../../python/scalelsd`) on both released checkpoints.

## Running it

```bash
# 1. Convert a checkpoint (writes model.safetensors + config.json)
python Scripts/convert.py -c scalelsd-vitbase-v1-train-sa1b.pt -o out/v1

# 2. Dump PyTorch per-stage activations for a fixed image
PYTHONPATH=<scalelsd repo> python Scripts/make_fixtures.py \
    -c scalelsd-vitbase-v1-train-sa1b.pt \
    -i assets/indoor.jpg -o Tests/MLXScaleLSDTests/Fixtures/v1

# 3. Compare, stage by stage
MLX_ENABLE_TF32=0 scalelsd parity -m out/v1 \
    -f Tests/MLXScaleLSDTests/Fixtures/v1/fixtures.safetensors
```

`MLX_ENABLE_TF32=0` is mandatory. MLX defaults it on, which runs fp32 matmuls at TF32
precision on the M-series matrix units and costs ~1e-3 relative — three orders of magnitude
above the real error.

Stages are compared in execution order, so the **first** row whose error jumps localises the
bug. `--detail` additionally reports interior-only error, which separates a padding bug
(error concentrated on the border) from a precision problem (error uniform).

## Results

Backbone, both checkpoints, 512×512 input — every stage:

| stage | v1 max rel | v2 max rel |
|---|---|---|
| stem → stage2 (ResNetV2) | 7.9e-07 … 2.7e-05 | same order |
| patch_proj, ViT blocks, taps | ~1.5e-05 … 2.9e-05 | ~1.4e-05 |
| DPT reassemble + refinenets | ~5e-06 … 1.3e-05 | ~7e-06 |
| **final 9-channel field** | **1.085e-05** | **1.364e-05** |

End-to-end detections on `assets/indoor.jpg`:

| | v1 | v2 |
|---|---|---|
| junctions matched within 0.01 px | 512/512 (100%) | 511/512 (99.80%) |
| lines matched within 0.01 px | 1879/1880 (99.95%) | 1581/1590 (99.43%) |
| segments scoring ≥ 10 | 544 (torch 544) | 597 (torch 594) |

(These use the fixture's preprocessed input, isolating the network and decoder. Running from the
source JPEG instead adds the preprocessing difference described below and shifts the ≥10 counts
by a couple of segments.)

## Why detections are not bit-exact

Two decode steps are chaotic — a change far below the field's own fp32 noise flips a discrete
choice. Both are properties of the algorithm, not defects in the port.

**The 512-junction cap.** Junctions are the top *N* heatmap peaks. The score gap at that
cutoff is `7.8e-05` for v1 but only **`5.96e-08`** for v2, while fp32 noise on a ~0.24
probability is ~1e-6. Which junction takes the last slot is therefore undetermined at fp32,
and v2 does swap exactly one. That single swap re-routes the handful of segments that had
snapped to it — hence 9 differing lines out of 1590.

**Nearest-junction assignment.** Every one of the 65 536 proposed segments snaps its endpoints
to the nearest junction by `argmin`. Where two junctions are near-equidistant from a proposal,
a 1e-5 perturbation flips the winner. This accounts for the single differing line in v1.

Parity is therefore gated on **agreement rate ≥ 99%**, not exact set equality, plus the
stage-wise field tolerance of 2e-4 which *is* deterministic. Comparing detections positionally
is meaningless: one extra or missing segment shifts every later element of a sorted list, and
a quantised match key reports spurious mismatches whenever a coordinate sits near a rounding
boundary (this produced a misleading "14 mismatches" before the metric was fixed).

## Preprocessing

The stage fixtures store the *already preprocessed* tensor, so the stage table above does not
exercise grayscale conversion or the resize at all. Pass `-i <image>` to `scalelsd parity` to
compare them explicitly — without that, a preprocessing regression is invisible. (One did slip
through exactly this way: a vDSP rewrite that folded `/255` into the BT.601 weights, changing
rounding.)

Swift preprocessing agrees with `cv2.imread(..., 0)` + `cv2.resize` to **1.4e-02** maximum
relative (1.2e-03 mean). That residual is structural, not a bug: the reference pipeline is 8-bit
end to end — `imread` quantises luma to `uint8`, then `resize` runs *fixed-point* bilinear and
quantises again — whereas this port carries float luma through a float resize using the same
`align_corners=False` sample mapping.

Reproducing the first of those quantisations was tried and rejected: rounding luma to 8 bits
before the resize moved agreement only from 1.406e-02 to 1.385e-02 (the residual is dominated by
the fixed-point *resize*, not the luma step) and moved detection counts slightly further from
the reference. The float pipeline is both more accurate and marginally closer, so it stands.

## The `--use-lsd` rectifier path

`scalelsd detect --use-lsd` replaces the network's predicted segment *direction* with one
derived from classical LSD, exactly as upstream does. Two measurements, because they fail
independently:

**The detector, in isolation.** Running [swift-lsd](../../swift-lsd) on the *reference*
preprocessed tensor — the same 8-bit image OpenCV sees — reproduces **510/514 segments within
0.01 px (99.22%)**, finding 514 where OpenCV finds 513. `scalelsd parity` reports this. The
handful of disagreements are greedy region-growing divergences, where two regions compete for a
shared boundary and a last-bit difference decides the split; the underlying edge is always
found, just carved differently.

**End to end from a JPEG.** On `assets/indoor.jpg` at score ≥ 10: 489 segments vs the
reference's 481 (v1), 509 vs 529 (v2) — a wider gap than the default path's 546/544 and
592/594. That is expected and attributable: LSD consumes the *quantised 8-bit* image directly
and is sensitive to its gradients, so the ~1.4e-02 preprocessing difference above perturbs which
segments it finds, and the chaotic decode then amplifies it. The isolated measurement is the one
that assesses the detector; this one compounds three sources.

## Gotchas found while porting

Each of these cost real debugging time; they are the reason the per-stage harness exists.

1. **`StdConv2dSame` eps is 1e-8, not the class default of 1e-6.** timm's hybrid ViT builds its
   ResNetV2 with `conv_layer=partial(StdConv2dSame, eps=1e-8)`. Using 1e-6 shifts the
   standardised weights enough to cost ~2e-3 relative error at the *very first stage*, growing
   to 2e-1 by the output.
2. **GroupNorm must use `pytorchCompatible: true`.** MLX's default groups channels with a
   different stride and produces plausible-but-wrong features (~0.5 relative error).
3. **SAME padding is asymmetric.** Only stride-1 odd kernels use static symmetric padding; the
   stem 7×7/s2, each stage's 3×3/s2 `conv2`, and the 3×3/s2 max pool need the trailing-biased
   dynamic pad (`(2,3)` and `(0,1)` respectively).
4. **`Upsample(scaleFactor:)` cannot hit an arbitrary target size.** It computes
   `Int(scale * dimension)`, so resizing the 24×24 position grid to 32×32 lands on 31.
   `bilinearResize(_:height:width:)` builds the sampling grid directly instead.
5. The refinenet upsamples use `align_corners=True`; the position-embedding resize uses
   `align_corners=False`. They are not interchangeable.

## Simplifications baked into the converter

Verified to reproduce upstream output to ~2e-6 relative (`Scripts/convert.py`):

- **Weight standardisation is precomputed.** `StdConv2dSame` re-standardises its weight on every
  forward; inference weights are frozen, so the standardised tensor is stored directly.
- **conv+BatchNorm pairs are folded** in `ResidualConvUnit_custom` (16 pairs).
- The 1000-class ImageNet head is dropped — ScaleLSD never calls it.

Parameter counts after conversion: v1 **122,525,833**, v2 **122,544,265**
(upstream minus the 769,000-parameter classifier and the 12,288-parameter net BatchNorm fold).
Both load with `verify: .all`.
