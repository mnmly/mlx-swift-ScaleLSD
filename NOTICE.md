# Third-party notices

## ScaleLSD (source)

This package is a Swift/MLX port of [ScaleLSD](https://github.com/ant-research/scalelsd)
by Zeran Ke, Bin Tan, Xianwei Zheng, Yujun Shen, Tianfu Wu and Nan Xue.

Upstream source is MIT-licensed, Copyright © 2023 Nan Xue — see
[`Scripts/licenses/ScaleLSD-MIT.txt`](Scripts/licenses/ScaleLSD-MIT.txt).

Files under `Sources/MLXScaleLSD/` carry `PORT FROM:` headers naming the upstream module each
was translated from. The architecture, weights and all model credit are the ScaleLSD authors';
this package contributes only the Swift/MLX translation.

## ScaleLSD (checkpoints)

The released checkpoints at [cherubicxn/scalelsd](https://huggingface.co/cherubicxn/scalelsd)
are Apache-2.0 — see [`Scripts/licenses/Apache-2.0.txt`](Scripts/licenses/Apache-2.0.txt).

Converted MLX copies are redistributed at
[mnmly/scalelsd-mlx](https://huggingface.co/mnmly/scalelsd-mlx) under that licence, with
attribution and a statement of the changes made (format conversion only — no retraining or
fine-tuning). `Scripts/convert.py` documents each transformation, and
[`docs/PARITY.md`](docs/PARITY.md) measures that they are numerically equivalent.

## timm

The backbone is `vit_base_r50_s16_384` from
[timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0, Copyright © Ross
Wightman). No timm code is vendored; its layer semantics — `StdConv2dSame` weight
standardisation, TF-style SAME padding, `GroupNormAct` — were reimplemented in Swift and
verified numerically against the Python originals.

## mlx-swift

[mlx-swift](https://github.com/ml-explore/mlx-swift) is MIT-licensed, Copyright © 2023-2024
Apple Inc. It is consumed as a SwiftPM dependency; no source is vendored.

## OpenCV / LSD

The `--use-lsd` rectifier path in upstream ScaleLSD calls OpenCV's
`createLineSegmentDetector`. OpenCV's implementation descends from the AGPL-licensed LSD
reference code by Rafael Grompone von Gioi, so **no OpenCV or LSD reference source is vendored
here**. That path is not implemented in this package; see the Status section of the README.
