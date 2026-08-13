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

Converted weights are published at
[**mnmly/scalelsd-mlx**](https://huggingface.co/mnmly/scalelsd-mlx) and download on first use:

```bash
scalelsd fetch --variant v2      # or from Swift: try await ModelStore.download(.v2)
```

The demo app's **Download Weights** button does the same thing.

To convert a checkpoint yourself instead — the upstream files are PyTorch pickles, which MLX
cannot read:

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

let directory = try await ModelStore.download(.v2)   // cached after the first call
let session = try ScaleLSDSession.load(directory: directory)
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

- `scalelsd fetch` — download converted weights into the local cache.
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

MacBook Pro (Apple M5 Max, 18 cores, 128 GB), `assets/indoor.jpg` at 512×512, Release build,
median of 20 runs:

| runtime | per image | vs this port |
|---|---|---|
| **mlx-swift (this port)** | **47 ms** | — |
| PyTorch 2.13, MPS | 88 ms | 1.9× slower |
| PyTorch 2.13, CPU | 652 ms | 14× slower |

Split: 39 ms preprocessing + forward, 9 ms wireframe decode. `scalelsd bench` reports the split
and confirms active memory is flat (467 MB, 0.0 MB growth over 20 iterations). The much larger
"peak" figure is MLX's reusable buffer cache, not a leak.

## Accuracy

Final HAT field matches PyTorch to **1.1e-05** (v1) / **1.4e-05** (v2) relative. End-to-end on
`assets/indoor.jpg`:

| | v1 | v2 |
|---|---|---|
| junctions within 0.01 px | 512/512 | 511/512 |
| segments within 0.01 px | 1879/1880 | 1581/1590 |

Detections are not bit-exact by construction: the 512-junction cap and the nearest-junction
assignment are discrete choices that a sub-noise perturbation can flip. Preprocessing also
differs from `cv2` by ~1.4% because the reference pipeline is 8-bit throughout.
[docs/PARITY.md](docs/PARITY.md) explains all three, with measurements.

## Status

Implemented and verified:

- DPT-hybrid backbone (ResNetV2 stages + ViT-B/16 with optional LayerScale), DPT reassemble /
  refinenet neck, 9-channel multitask head.
- HAT-field decode, junction NMS + top-k, wireframe matching.
- `encodels` line-to-field encoder (the CUDA extension upstream), matching a scalar reference
  to 1.2e-07.
- The `--use-lsd` rectifier path, via [swift-lsd](../swift-lsd) — an independent implementation
  of the LSD algorithm matching `cv2.createLineSegmentDetector(LSD_REFINE_NONE)`. Only the
  *direction* channel of the LSD field survives into the network's output (upstream overwrites
  the rest), so that is all this port computes; see
  `Sources/MLXScaleLSD/Decode/LineFieldEncoder.swift`.
- Shared `ScaleLSDSession`, CLI, SwiftUI demo, benchmark, parity harness.

## Documentation

`MLXScaleLSD` ships DocC reference docs for every public symbol:

```bash
./Scripts/build_docs.sh            # static site into docs/MLXScaleLSD/
./Scripts/build_docs.sh preview    # local server with live reload
```

`docs/PARITY.md` is the hand-written companion covering verification methodology.

## License

This port is MIT-licensed. Upstream ScaleLSD source is MIT (Copyright © 2023 Nan Xue) — see
`LICENSE.upstream-scalelsd`. The checkpoints are Apache-2.0, and the converted copies at
[mnmly/scalelsd-mlx](https://huggingface.co/mnmly/scalelsd-mlx) are redistributed under that
licence with attribution and a statement of the conversion changes. All model credit belongs to
the ScaleLSD authors.

```bibtex
@inproceedings{ScaleLSD,
    title = {ScaleLSD: Scalable Deep Line Segment Detection Streamlined},
    author = {Zeran Ke and Bin Tan and Xianwei Zheng and Yujun Shen and Tianfu Wu and Nan Xue},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2025},
}
```
