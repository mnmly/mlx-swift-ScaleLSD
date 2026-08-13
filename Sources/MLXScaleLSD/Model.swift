// Top-level ScaleLSD network: DPT-hybrid backbone producing a 9-channel HAT field.
// PORT FROM: scalelsd/ssl/backbones/dpt/models.py (DPT, DPTFieldModel)
//            scalelsd/ssl/models/detector.py (ScaleLSD.forward_backbone)

import Foundation
import MLX
import MLXNN

/// DPT encoder: the hybrid ViT plus the two reassemble levels that follow it.
///
/// Named `pretrained` to match the upstream checkpoint namespace.
final class DPTEncoder: Module {

    let model: HybridVisionTransformer
    let reassemble3: Reassemble
    let reassemble4: Reassemble

    /// ViT blocks tapped for DPT levels 3 and 4 (upstream `hooks = [0, 1, 8, 11]`; the first
    /// two index ResNet stages rather than blocks).
    let tapLayers = [8, 11]

    init(configuration: ScaleLSDConfiguration, levelChannels: [Int]) {
        self.model = HybridVisionTransformer(configuration: configuration)
        self.reassemble3 = Reassemble(
            dimensions: configuration.vitFeatures, outputChannels: levelChannels[2],
            downsample: false)
        self.reassemble4 = Reassemble(
            dimensions: configuration.vitFeatures, outputChannels: levelChannels[3],
            downsample: true)
        super.init()
    }

    /// - Returns: the four DPT levels, finest (stride 4) to coarsest (stride 32).
    func callAsFunction(_ x: MLXArray, recorder: TraceRecorder? = nil) -> [MLXArray] {
        let gridHeight = x.dim(1) / model.patchSize
        let gridWidth = x.dim(2) / model.patchSize

        let (layer1, layer2, taps) = model(x, tapLayers: tapLayers, recorder: recorder)
        let layer3 = reassemble3(taps[0], gridHeight: gridHeight, gridWidth: gridWidth)
        let layer4 = reassemble4(taps[1], gridHeight: gridHeight, gridWidth: gridWidth)
        recorder?.record("layer3", layer3)
        recorder?.record("layer4", layer4)

        return [layer1, layer2, layer3, layer4]
    }
}

/// The DPT field model: encoder + refinement decoder + multitask head.
final class DPTFieldModel: Module {

    let pretrained: DPTEncoder
    let scratch: Scratch

    init(configuration: ScaleLSDConfiguration) {
        // Hybrid backbone level widths: two ResNet stages, then two ViT taps.
        let levelChannels = [256, 512, 768, 768]
        self.pretrained = DPTEncoder(
            configuration: configuration, levelChannels: levelChannels)
        self.scratch = Scratch(
            inputChannels: levelChannels, features: configuration.features,
            headSize: configuration.headSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { scratch(pretrained(x)) }
}

/// ScaleLSD: a line-segment detector predicting a dense HAT (Holistic Attraction) field.
///
/// The network is a thin wrapper over the DPT backbone; all detection logic lives in the
/// decoder (``HATField`` and ``WireframeDecoder``). Use ``ScaleLSDSession`` rather than
/// driving this type directly.
public final class ScaleLSD: Module {

    let backbone: DPTFieldModel

    /// Static description of the checkpoint this instance was built for.
    public let configuration: ScaleLSDConfiguration

    public init(configuration: ScaleLSDConfiguration) {
        self.configuration = configuration
        self.backbone = DPTFieldModel(configuration: configuration)
        super.init()
    }

    /// Run the backbone.
    ///
    /// - Parameter x: NHWC image batch, values in `0...1`. A single-channel input is
    ///   replicated to three channels, matching `DPTFieldModel.forward`.
    /// - Returns: the raw `(B, H/2, W/2, 9)` HAT field, before any activation.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        if input.dim(-1) == 1 {
            input = concatenated([input, input, input], axis: -1)
        }
        return backbone(input)
    }
}
