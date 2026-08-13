# mlx-swift-ScaleLSD

Inference-only Swift port of [ScaleLSD](https://github.com/ant-research/scalelsd)
(scalable deep line-segment detection) on top of
[mlx-swift](https://github.com/ml-explore/mlx-swift).

Python reference lives at `../../python/scalelsd`.

## Build and test

`swift test` cannot be used — it fails to load mlx-swift's `default.metallib` from its resource
bundle. Always:

```bash
swift build -c release                                                  # library + CLI
xcodebuild -scheme mlx-swift-ScaleLSD-Package -destination 'platform=macOS' test
```

Confirm the scheme name with `xcodebuild -list` rather than guessing it.

## Architecture invariants

- **One shared driver.** `ScaleLSDSession` owns weight loading, preprocessing, inference and
  post-processing. The `scalelsd` CLI and `Examples/ScaleLSDDemo` consume *only* it, so they
  cannot drift. New workload code goes in the library, never in a frontend. Cadence (log
  intervals, `autoreleasepool`, MainActor hops) stays frontend-side.
- **The session takes pixels, never paths it opens itself.** A sandboxed caller must bake a
  `CGImage` while its file grant is open — lazy decode otherwise faults after the grant expires.
- **Cheap post-processing is separable from the forward pass.** The raw 9-channel HAT field and
  the decoded candidate set are retained on the result; changing the score threshold or the
  junction-heatmap threshold must re-filter in place, never re-run the network.
- **Weight loading verifies with `.all`** (`noUnusedKeys` + `allModelKeysSet` + `shapeMismatch`).
  Do not relax it; it is what catches converter key-mapping mistakes.
- **Derived constants must not become parameters.** A stored `MLXArray` on a `Module` is
  registered as a checkpoint parameter by reflection. Wrap coordinate grids and index vectors so
  they are not picked up by `parameters()`.

## The two checkpoints

`scalelsd-vitbase-v1-train-sa1b.pt` and `-v2-`. The *only* architectural difference is
LayerScale: v2 carries `blocks.N.ls1.gamma` / `ls2.gamma` (24 extra tensors), v1 does not.
Upstream keys off the substring `v1` in the filename (`load_scalelsd_model`); this port records
it explicitly in the converted `config.json` instead.

## Parity

Numerical parity against PyTorch is verified per stage — see `docs/PARITY.md`. MLX defaults
`MLX_ENABLE_TF32=1`, so fp32 matmuls run at TF32 on M-series matrix units (~1e-3 relative).
Parity runs must set it to `0`.

Two upstream details bite:

- The ResNetV2 stem/stages use **StdConv2dSame**: weight-standardised convolution with TF-style
  *asymmetric* SAME padding. Getting the padding split wrong shifts every downstream feature.
- `_resize_pos_embed` bilinearly resizes the 24×24 pretrained position grid to H/16 × W/16
  **without** `align_corners`, and keeps the cls-token row untouched.


## Documentation

`MLXScaleLSD` ships DocC-generated reference docs (`Sources/MLXScaleLSD/Documentation.docc/`,
built by `Scripts/build_docs.sh`). **`///` comments on public/`open` symbols are published**, so
treat them as shipped surface rather than internal hints.

When you add or modify a `public` or `open` declaration:

- Write a `///` comment. One-sentence summary, then a paragraph when the *why* is non-obvious.
  Don't restate what the signature already says.
- Document each parameter with `- Parameter name:` — use the **internal** name when there is an
  external label (`- key:`, not `- forKey:`), or DocC warns.
- Cross-reference with double-backticks, e.g. ``` ``ScaleLSDSession/analyze(_:options:)`` ```.
  The syntax is signature-sensitive: `foo(_:)` and `foo(_:_:)` are different symbols.
- File new top-level symbols under a `## Topics` group in
  `Sources/MLXScaleLSD/Documentation.docc/MLXScaleLSD.md`. Groups are organised by *user task*;
  there is deliberately no uncurated "Classes" bucket, so an unfiled type shows up as a warning.
- Keep the `PORT FROM:` header comments accurate — they are the map back to the Python source.

Verify with `./Scripts/build_docs.sh` and expect no warnings on user-authored prose.
