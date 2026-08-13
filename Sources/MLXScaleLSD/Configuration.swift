// PORT FROM: scalelsd/ssl/backbones/build.py, scalelsd/ssl/models/detector.py

import Foundation

/// Static description of a ScaleLSD checkpoint.
///
/// ScaleLSD ships two checkpoints whose architectures differ in exactly one respect:
/// v2 applies LayerScale inside every ViT block, v1 does not. Upstream infers this from
/// the substring `v1` in the checkpoint filename; this port records it explicitly in the
/// `config.json` written by `Scripts/convert.py`.
public struct ScaleLSDConfiguration: Codable, Sendable, Equatable {

    /// Whether the ViT blocks carry `ls1`/`ls2` LayerScale parameters (v2 checkpoints).
    public var useLayerScale: Bool

    /// ViT embedding width.
    public var vitFeatures: Int

    /// Number of ViT blocks.
    public var vitDepth: Int

    /// Attention heads per ViT block.
    public var vitHeads: Int

    /// Effective patch size of the hybrid embedding (ResNet stride 16 × 1×1 projection).
    public var patchSize: Int

    /// Channel width shared by the DPT `scratch` convolutions and refinement blocks.
    public var features: Int

    /// Output channels of each of the five multitask heads; sums to 9.
    public var headSize: [Int]

    /// Ratio of input resolution to output resolution of the HAT field.
    public var stride: Int

    /// Distance normalisation used when decoding the HAT distance field, in output pixels.
    public var distanceThreshold: Float

    enum CodingKeys: String, CodingKey {
        case useLayerScale = "use_layer_scale"
        case vitFeatures = "vit_features"
        case vitDepth = "vit_depth"
        case vitHeads = "vit_heads"
        case patchSize = "patch_size"
        case features
        case headSize = "head_size"
        case stride
        case distanceThreshold = "distance_threshold"
    }

    public init(
        useLayerScale: Bool,
        vitFeatures: Int = 768,
        vitDepth: Int = 12,
        vitHeads: Int = 12,
        patchSize: Int = 16,
        features: Int = 256,
        headSize: [Int] = [3, 1, 1, 2, 2],
        stride: Int = 2,
        distanceThreshold: Float = 5.0
    ) {
        self.useLayerScale = useLayerScale
        self.vitFeatures = vitFeatures
        self.vitDepth = vitDepth
        self.vitHeads = vitHeads
        self.patchSize = patchSize
        self.features = features
        self.headSize = headSize
        self.stride = stride
        self.distanceThreshold = distanceThreshold
    }

    /// The v1 checkpoint (`scalelsd-vitbase-v1-train-sa1b`) — no LayerScale.
    public static let v1 = ScaleLSDConfiguration(useLayerScale: false)

    /// The v2 checkpoint (`scalelsd-vitbase-v2-train-sa1b`) — LayerScale in every block.
    public static let v2 = ScaleLSDConfiguration(useLayerScale: true)

    /// Total channels emitted by the multitask head (3 angle + 1 distance + 1 residual
    /// + 2 junction-location logits + 2 junction offsets).
    public var outputChannels: Int { headSize.reduce(0, +) }

    /// Decode a `config.json` produced by `Scripts/convert.py`.
    public static func load(contentsOf url: URL) throws -> ScaleLSDConfiguration {
        try JSONDecoder().decode(ScaleLSDConfiguration.self, from: Data(contentsOf: url))
    }
}
