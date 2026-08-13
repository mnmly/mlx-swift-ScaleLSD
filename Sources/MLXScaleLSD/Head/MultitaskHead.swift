// The 9-channel HAT-field prediction head.
// PORT FROM: scalelsd/ssl/backbones/multi_task_head.py (MultitaskHead)
//            scalelsd/ssl/backbones/dpt/models.py (DPTFieldModel head)
//
// Upstream nests two `nn.Sequential`s, giving positional keys like
// `output_conv.0.weight` and `output_conv.2.heads.3.0.weight`. Scripts/convert.py renames
// these to `output_conv.stem.*` and `output_conv.heads.<i>.conv{1,2}.*`.

import Foundation
import MLX
import MLXNN

/// One task branch: 3×3 reduction, ReLU, then a 1×1 projection to that task's channels.
final class MultitaskBranch: Module {

    let conv1: Conv2d
    let conv2: Conv2d

    init(inputChannels: Int, hiddenChannels: Int, outputChannels: Int) {
        self.conv1 = Conv2d(
            inputChannels: inputChannels, outputChannels: hiddenChannels, kernelSize: .init(3),
            padding: .init(1), bias: true)
        self.conv2 = Conv2d(
            inputChannels: hiddenChannels, outputChannels: outputChannels, kernelSize: .init(1),
            bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(relu(conv1(x))) }
}

/// Final head: a shared 3×3 stem then five parallel branches concatenated on the channel axis.
///
/// The 9 output channels are consumed positionally by the decoder as
/// `[0..<3]` HAT angles, `[3]` distance, `[4]` residual, `[5..<7]` junction logits,
/// `[7..<9]` junction sub-pixel offsets.
final class MultitaskOutput: Module {

    /// The `Conv2d(features, features/2, 3)` that precedes the branches.
    let stem: Conv2d
    let heads: [MultitaskBranch]

    init(inputChannels: Int, headSize: [Int]) {
        let hidden = inputChannels / 2
        self.stem = Conv2d(
            inputChannels: inputChannels, outputChannels: hidden, kernelSize: .init(3),
            padding: .init(1), bias: true)
        self.heads = headSize.map {
            MultitaskBranch(
                inputChannels: hidden, hiddenChannels: hidden / 4, outputChannels: $0)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shared = relu(stem(x))
        return concatenated(heads.map { $0(shared) }, axis: -1)
    }
}
