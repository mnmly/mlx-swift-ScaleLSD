// The ResNetV2 half of timm's `vit_base_r50_s16_384` hybrid embedding.
// PORT FROM: timm/models/resnetv2.py (Bottleneck, DownsampleConv, create_resnetv2_stem)
//
// Non-preactivation bottleneck ("v1.5") blocks with GroupNorm(32) and weight-standardised
// SAME convolutions. Weight standardisation is baked into the checkpoint by
// Scripts/convert.py, so `SameConv2d` here is a plain convolution.
//
// Property names deliberately match the upstream state-dict keys exactly (`conv1`, `norm1`,
// `downsample`, `stem`, `stages`, `blocks`), so no `@ModuleInfo(key:)` remapping is needed.

import Foundation
import MLX
import MLXNN

/// GroupNorm shared by every ResNetV2 norm site: 32 groups, eps 1e-5, PyTorch grouping.
///
/// `pytorchCompatible` is required — MLX's default groups channels with a different stride
/// and would silently produce plausible-but-wrong features.
private func groupNorm(_ channels: Int) -> GroupNorm {
    GroupNorm(
        groupCount: 32, dimensions: channels, eps: 1e-5, affine: true, pytorchCompatible: true)
}

/// 1×1 projection on the residual shortcut, present only on the first block of each stage.
final class DownsampleConv: Module {

    let conv: SameConv2d
    let norm: GroupNorm

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        self.conv = SameConv2d(
            inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1,
            stride: stride)
        self.norm = groupNorm(outputChannels)
        super.init()
    }

    /// No activation — upstream builds this norm with `apply_act=False`.
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(conv(x)) }
}

/// Non-preactivation bottleneck block.
final class Bottleneck: Module {

    /// `nil` for every block except the first of a stage; a nil submodule contributes no
    /// checkpoint keys, so strict weight verification still passes.
    let downsample: DownsampleConv?

    let conv1: SameConv2d
    let norm1: GroupNorm
    let conv2: SameConv2d
    let norm2: GroupNorm
    let conv3: SameConv2d
    let norm3: GroupNorm

    init(inputChannels: Int, outputChannels: Int, stride: Int, useDownsample: Bool) {
        let mid = outputChannels / 4
        self.downsample =
            useDownsample
            ? DownsampleConv(
                inputChannels: inputChannels, outputChannels: outputChannels, stride: stride)
            : nil
        self.conv1 = SameConv2d(
            inputChannels: inputChannels, outputChannels: mid, kernelSize: 1)
        self.norm1 = groupNorm(mid)
        // The stage's spatial reduction happens here, not on conv1.
        self.conv2 = SameConv2d(
            inputChannels: mid, outputChannels: mid, kernelSize: 3, stride: stride)
        self.norm2 = groupNorm(mid)
        self.conv3 = SameConv2d(
            inputChannels: mid, outputChannels: outputChannels, kernelSize: 1)
        self.norm3 = groupNorm(outputChannels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = downsample?(x) ?? x
        var h = relu(norm1(conv1(x)))
        h = relu(norm2(conv2(h)))
        h = norm3(conv3(h))  // norm3 carries no activation
        return relu(h + shortcut)
    }
}

/// One resolution stage: a run of bottleneck blocks, the first of which downsamples.
final class ResNetStage: Module {

    let blocks: [Bottleneck]

    init(inputChannels: Int, outputChannels: Int, stride: Int, count: Int) {
        self.blocks = (0 ..< count).map { index in
            Bottleneck(
                inputChannels: index == 0 ? inputChannels : outputChannels,
                outputChannels: outputChannels,
                stride: index == 0 ? stride : 1,
                useDownsample: index == 0)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks { h = block(h) }
        return h
    }
}

/// 7×7 stride-2 convolution, norm+ReLU, then a SAME 3×3 stride-2 max pool (total stride 4).
final class ResNetStem: Module {

    let conv: SameConv2d
    let norm: GroupNorm
    /// Parameter-free, so it contributes no checkpoint keys.
    private let pool: MaxPool2dSame

    init(inputChannels: Int = 3, outputChannels: Int = 64) {
        self.conv = SameConv2d(
            inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 7,
            stride: 2)
        self.norm = groupNorm(outputChannels)
        self.pool = MaxPool2dSame(kernelSize: 3, stride: 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { pool(relu(norm(conv(x)))) }
}

/// The 3-stage ResNetV2 trunk feeding the ViT's 1×1 patch projection.
///
/// Emits the two high-resolution DPT taps alongside its final output: upstream captures
/// `stages[0]` and `stages[1]` with forward hooks, which this port returns explicitly.
final class ResNetV2Backbone: Module {

    let stem: ResNetStem
    let stages: [ResNetStage]

    override init() {
        self.stem = ResNetStem()
        self.stages = [
            // (in, out, stride, blocks) — strides give total reductions of 4, 8 and 16.
            ResNetStage(inputChannels: 64, outputChannels: 256, stride: 1, count: 3),
            ResNetStage(inputChannels: 256, outputChannels: 512, stride: 2, count: 4),
            ResNetStage(inputChannels: 512, outputChannels: 1024, stride: 2, count: 9),
        ]
        super.init()
    }

    /// - Returns: `stage0` (stride 4, 256ch) and `stage1` (stride 8, 512ch) as the DPT taps,
    ///   plus `output` (stride 16, 1024ch) for the patch projection.
    func callAsFunction(_ x: MLXArray, recorder: TraceRecorder? = nil) -> (
        stage0: MLXArray, stage1: MLXArray, output: MLXArray
    ) {
        let h = stem(x)
        recorder?.record("stem", h)
        let s0 = stages[0](h)
        recorder?.record("stage0", s0)
        let s1 = stages[1](s0)
        recorder?.record("stage1", s1)
        let s2 = stages[2](s1)
        recorder?.record("stage2", s2)
        return (s0, s1, s2)
    }
}
