# mlx-swift-ScaleLSD

[ScaleLSD](https://github.com/ant-research/scalelsd) — scalable deep line-segment detection —
ported to Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).
Inference only.

The port reproduces the reference graph stage by stage and is verified against PyTorch on both
released checkpoints: see [docs/PARITY.md](docs/PARITY.md).

## Install

```swift
.package(url: "https://github.com/mnmly/mlx-swift-ScaleLSD", from: "0.1.0"),
```

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "MLXScaleLSD", package: "mlx-swift-ScaleLSD"),
    ]
)
```

## Weights

The upstream checkpoints are PyTorch pickles, which MLX cannot read. Convert one first:

```bash
# grab a checkpoint from https://huggingface.co/cherubicxn/scalelsd
uv run --with torch --with safetensors python Scripts/convert.py \
    -c scalelsd-vitbase-v2-train-sa1b.pt -o models/scalelsd-vitbase-v2
```

That writes `config.json` + `model.safetensors`. Both released checkpoints are supported; the
only architectural difference is LayerScale, which the converter records in `config.json`:

| checkpoint | LayerScale | parameters |
|---|---|---|
| `scalelsd-vitbase-v1-train-sa1b` | no | 122,525,833 |
| `scalelsd-vitbase-v2-train-sa1b` | yes | 122,544,265 |

## Quick start

```swift
import MLXScaleLSD

let session = try ScaleLSDSession.load(directory: modelDirectory)
let image = try ScaleLSDSession.loadImage(at: imageURL)
let result = try session.detect(image)

for segment in result.segments(minimumScore: 10) {
    print(segment.x1, segment.y1, segment.x2, segment.y2, segment.score)
}
```

`DetectionResult` holds *every* candidate segment; `segments(minimumScore:)` re-filters in
place, so a threshold slider never re-runs the network. To change junction thresholds without
re-running the backbone, keep the field and re-decode:

```swift
let field = try session.analyze(image)          // expensive: the forward pass
let loose = session.decode(field, options: .init(junctionThreshold: 0.004))
let tight = session.decode(field, options: .init(junctionThreshold: 0.05, useNMS: true))
```

## Command line

```bash
xcodebuild -scheme scalelsd -destination 'platform=macOS' \
    -configuration Release -derivedDataPath .xcdd build

.xcdd/Build/Products/Release/scalelsd detect \
    -m models/scalelsd-vitbase-v2 -i image.jpg -e png --save-to out/
```

`detect` mirrors `predictor/predict.py`: `--threshold`, `--junction-hm`, `--num-junctions`,
`--use-nms`, and `--ext png|json`. There are two more subcommands:

- `scalelsd bench` — throughput plus a leak check that watches `MLX.Memory.activeMemory`.
- `scalelsd parity` — stage-by-stage comparison against PyTorch fixtures.

## Demo app

A SwiftUI demo ships as an executable target — drop an image, drag the thresholds, download
weights from within the app:

```bash
xcodebuild -scheme ScaleLSDDemo -destination 'platform=macOS' \
    -configuration Release -derivedDataPath .xcdd build
.xcdd/Build/Products/Release/ScaleLSDDemo
```

Both frontends consume only `ScaleLSDSession`, so they cannot drift.

## Performance

MacBook Pro (M2 Max), `assets/indoor.jpg` at 512×512, Release build, median of 20 runs:

| runtime | per image | vs this port |
|---|---|---|
| **mlx-swift (this port)** | **57 ms** | — |
| PyTorch 2.13, MPS | 88 ms | 1.5× slower |
| PyTorch 2.13, CPU | 652 ms | 11× slower |

`scalelsd bench` also confirms active memory is flat (467.4 MB, 0.0 MB growth over 20
iterations). The much larger "peak" figure is MLX's reusable buffer cache, not a leak.

## Accuracy

Final HAT field matches PyTorch to **1.1e-05** (v1) / **1.4e-05** (v2) relative. End-to-end on
`assets/indoor.jpg`:

| | v1 | v2 |
|---|---|---|
| junctions within 0.01 px | 512/512 | 511/512 |
| segments within 0.01 px | 1879/1880 | 1581/1590 |

Detections are not bit-exact by construction: the 512-junction cap and the nearest-junction
assignment are discrete choices that a sub-noise perturbation can flip. [docs/PARITY.md](docs/PARITY.md)
explains both, with measurements.

## Status

Implemented and verified:

- DPT-hybrid backbone (ResNetV2 stages + ViT-B/16 with optional LayerScale), DPT reassemble /
  refinenet neck, 9-channel multitask head.
- HAT-field decode, junction NMS + top-k, wireframe matching.
- `encodels` line-to-field encoder (the CUDA extension upstream), matching a scalar reference
  to 1.2e-07.
- Shared `ScaleLSDSession`, CLI, SwiftUI demo, benchmark, parity harness.

Not yet implemented:

- **The `--use-lsd` rectifier path.** It needs a line-segment detector equivalent to OpenCV's
  `createLineSegmentDetector(LSD_REFINE_NONE)`, which is a self-contained image-processing
  algorithm rather than a model, and is intended to live in its own package. The ScaleLSD-side
  half of that path — the field encoder the LSD segments feed — is already ported and verified.
  Note that only the *direction* channel of the LSD field survives into the network's output;
  see the discussion in `Sources/MLXScaleLSD/Decode/LineFieldEncoder.swift`.

## License

This port is MIT-licensed. Upstream ScaleLSD is MIT — see `LICENSE.upstream-scalelsd`.

```bibtex
@inproceedings{ScaleLSD,
    title = {ScaleLSD: Scalable Deep Line Segment Detection Streamlined},
    author = {Zeran Ke and Bin Tan and Xianwei Zheng and Yujun Shen and Tianfu Wu and Nan Xue},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2025},
}
```
