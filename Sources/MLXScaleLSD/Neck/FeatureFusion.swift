// DPT refinement path: `scratch` projections, feature-fusion blocks, and the fusion decoder.
// PORT FROM: scalelsd/ssl/backbones/dpt/blocks.py
//            (_make_scratch, ResidualConvUnit_custom, FeatureFusionBlock_custom)
//
// Upstream's `ResidualConvUnit_custom` runs conv(bias=False) -> BatchNorm2d. Scripts/convert.py
// folds each pair into a single biased convolution, so no BatchNorm appears here.

import Foundation
import MLX
import MLXNN

/// Pre-activation residual unit: `x + conv2(relu(conv1(relu(x))))`.
final class ResidualConvUnit: Module {

    let conv1: Conv2d
    let conv2: Conv2d

    init(features: Int) {
        // bias comes from the folded BatchNorm.
        self.conv1 = Conv2d(
            inputChannels: features, outputChannels: features, kernelSize: .init(3),
            padding: .init(1), bias: true)
        self.conv2 = Conv2d(
            inputChannels: features, outputChannels: features, kernelSize: .init(3),
            padding: .init(1), bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(relu(x))
        h = conv2(relu(h))
        return h + x
    }
}

/// Fuse a coarser path with a skip connection, upsample ×2, then project.
///
/// `resConfUnit1` is only exercised when a skip feature is supplied — `refinenet4` is called
/// with a single argument. The unit is still allocated because the checkpoint contains its
/// weights for all four blocks.
final class FeatureFusionBlock: Module {

    let resConfUnit1: ResidualConvUnit
    let resConfUnit2: ResidualConvUnit
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    /// Upstream builds these blocks with `align_corners=True`.
    private let upsample = Upsample(scaleFactor: .float(2), mode: .linear(alignCorners: true))

    init(features: Int) {
        self.resConfUnit1 = ResidualConvUnit(features: features)
        self.resConfUnit2 = ResidualConvUnit(features: features)
        self._outConv.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features, kernelSize: .init(1), bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray? = nil) -> MLXArray {
        var output = x
        if let skip {
            output = output + resConfUnit1(skip)
        }
        output = resConfUnit2(output)
        return outConv(upsample(output))
    }
}

/// The DPT decoder: 3×3 projections of the four taps, then four fusion stages.
final class Scratch: Module {

    @ModuleInfo(key: "layer1_rn") var layer1RN: Conv2d
    @ModuleInfo(key: "layer2_rn") var layer2RN: Conv2d
    @ModuleInfo(key: "layer3_rn") var layer3RN: Conv2d
    @ModuleInfo(key: "layer4_rn") var layer4RN: Conv2d

    let refinenet1: FeatureFusionBlock
    let refinenet2: FeatureFusionBlock
    let refinenet3: FeatureFusionBlock
    let refinenet4: FeatureFusionBlock

    @ModuleInfo(key: "output_conv") var outputConv: MultitaskOutput

    init(inputChannels: [Int], features: Int, headSize: [Int]) {
        func projection(_ channels: Int) -> Conv2d {
            Conv2d(
                inputChannels: channels, outputChannels: features, kernelSize: .init(3),
                padding: .init(1), bias: false)
        }
        self._layer1RN.wrappedValue = projection(inputChannels[0])
        self._layer2RN.wrappedValue = projection(inputChannels[1])
        self._layer3RN.wrappedValue = projection(inputChannels[2])
        self._layer4RN.wrappedValue = projection(inputChannels[3])

        self.refinenet1 = FeatureFusionBlock(features: features)
        self.refinenet2 = FeatureFusionBlock(features: features)
        self.refinenet3 = FeatureFusionBlock(features: features)
        self.refinenet4 = FeatureFusionBlock(features: features)

        self._outputConv.wrappedValue = MultitaskOutput(
            inputChannels: features, headSize: headSize)
        super.init()
    }

    /// - Parameter layers: the four DPT taps, finest (stride 4) to coarsest (stride 32).
    func callAsFunction(_ layers: [MLXArray], recorder: TraceRecorder? = nil) -> MLXArray {
        let l1 = layer1RN(layers[0])
        let l2 = layer2RN(layers[1])
        let l3 = layer3RN(layers[2])
        let l4 = layer4RN(layers[3])
        recorder?.record("layer1_rn", l1)
        recorder?.record("layer2_rn", l2)
        recorder?.record("layer3_rn", l3)
        recorder?.record("layer4_rn", l4)

        let path4 = refinenet4(l4)
        let path3 = refinenet3(path4, skip: l3)
        let path2 = refinenet2(path3, skip: l2)
        let path1 = refinenet1(path2, skip: l1)
        recorder?.record("path4", path4)
        recorder?.record("path3", path3)
        recorder?.record("path2", path2)
        recorder?.record("path1", path1)

        return outputConv(path1)
    }
}
