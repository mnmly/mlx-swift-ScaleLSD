// The ViT half of timm's `vit_base_r50_s16_384` hybrid, plus DPT's `forward_flex` override.
// PORT FROM: timm/models/vision_transformer.py and
//            scalelsd/ssl/backbones/dpt/vit.py (forward_flex, _resize_pos_embed)

import Foundation
import MLX
import MLXNN

/// timm's `LayerScale`: a per-channel gain on each residual branch.
///
/// Present only in v2 checkpoints. The parameter is named `gamma` upstream.
final class LayerScale: Module {

    let gamma: MLXArray

    init(dimensions: Int) {
        self.gamma = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

/// Standard ViT multi-head self-attention with a fused QKV projection.
final class ViTAttention: Module {

    let qkv: Linear
    let proj: Linear

    let numHeads: Int
    let scale: Float

    init(dimensions: Int, numHeads: Int) {
        self.numHeads = numHeads
        // `sqrt` in module scope resolves to MLX's array version; force the scalar one.
        self.scale = 1.0 / Float(dimensions / numHeads).squareRoot()
        self.qkv = Linear(dimensions, dimensions * 3, bias: true)
        self.proj = Linear(dimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, tokens, channels) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = channels / numHeads

        // (B, N, 3C) -> (3, B, heads, N, headDim)
        let fused = qkv(x)
            .reshaped(batch, tokens, 3, numHeads, headDim)
            .transposed(2, 0, 3, 1, 4)
        let queries = fused[0]
        let keys = fused[1]
        let values = fused[2]

        let attended = scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)

        return proj(attended.transposed(0, 2, 1, 3).reshaped(batch, tokens, channels))
    }
}

/// ViT feed-forward block. Upstream uses exact (non-tanh) GELU.
final class ViTMLP: Module {

    let fc1: Linear
    let fc2: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self.fc1 = Linear(dimensions, hiddenDimensions, bias: true)
        self.fc2 = Linear(hiddenDimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// One pre-norm transformer block, optionally with LayerScale on both residual branches.
final class ViTBlock: Module {

    let norm1: LayerNorm
    let attn: ViTAttention
    /// `nil` on v1 checkpoints, which carry no `ls1`/`ls2` tensors.
    let ls1: LayerScale?
    let norm2: LayerNorm
    let mlp: ViTMLP
    let ls2: LayerScale?

    init(dimensions: Int, numHeads: Int, mlpRatio: Float = 4.0, useLayerScale: Bool) {
        self.norm1 = LayerNorm(dimensions: dimensions, eps: 1e-6)
        self.attn = ViTAttention(dimensions: dimensions, numHeads: numHeads)
        self.ls1 = useLayerScale ? LayerScale(dimensions: dimensions) : nil
        self.norm2 = LayerNorm(dimensions: dimensions, eps: 1e-6)
        self.mlp = ViTMLP(
            dimensions: dimensions, hiddenDimensions: Int(Float(dimensions) * mlpRatio))
        self.ls2 = useLayerScale ? LayerScale(dimensions: dimensions) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = attn(norm1(x))
        if let ls1 { h = ls1(h) }
        var out = x + h

        var m = mlp(norm2(out))
        if let ls2 { m = ls2(m) }
        out = out + m
        return out
    }
}

/// The hybrid patch embedding: ResNetV2 trunk followed by a 1×1 projection to ViT width.
final class HybridPatchEmbed: Module {

    let backbone: ResNetV2Backbone
    let proj: Conv2d

    init(dimensions: Int, backboneChannels: Int = 1024) {
        self.backbone = ResNetV2Backbone()
        self.proj = Conv2d(
            inputChannels: backboneChannels, outputChannels: dimensions, kernelSize: .init(1),
            bias: true)
        super.init()
    }
}

/// Hybrid ViT with DPT's flexible-resolution forward.
///
/// Differs from a stock timm forward in two ways, both from `scalelsd/ssl/backbones/dpt/vit.py`:
/// the position embedding is resampled to the actual token grid rather than requiring 384×384
/// input, and the intermediate activations DPT taps are returned instead of being captured by
/// forward hooks.
final class HybridVisionTransformer: Module {

    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray

    @ModuleInfo(key: "patch_embed") var patchEmbed: HybridPatchEmbed
    let blocks: [ViTBlock]
    let norm: LayerNorm

    let patchSize: Int
    /// Number of leading non-spatial tokens (the class token).
    let startIndex = 1

    init(configuration: ScaleLSDConfiguration) {
        let dim = configuration.vitFeatures
        self.patchSize = configuration.patchSize
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, dim])
        // 24×24 pretrained grid (384/16) plus the class token.
        self._posEmbed.wrappedValue = MLXArray.zeros([1, 577, dim])
        self._patchEmbed.wrappedValue = HybridPatchEmbed(dimensions: dim)
        self.blocks = (0 ..< configuration.vitDepth).map { _ in
            ViTBlock(
                dimensions: dim, numHeads: configuration.vitHeads,
                useLayerScale: configuration.useLayerScale)
        }
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        super.init()
    }

    /// Resample the pretrained position grid to `gridHeight × gridWidth`, leaving the
    /// class-token row untouched.
    private func resizedPositionEmbedding(gridHeight: Int, gridWidth: Int) -> MLXArray {
        let tokenPart = posEmbed[0..., ..<startIndex]
        let gridPart = posEmbed[0, startIndex...]

        let side = Int(Double(gridPart.dim(0)).squareRoot())
        if side == gridHeight && side == gridWidth {
            return posEmbed
        }

        let dim = gridPart.dim(-1)
        let grid = gridPart.reshaped(1, side, side, dim)
        let resized = bilinearResize(grid, height: gridHeight, width: gridWidth)
            .reshaped(1, gridHeight * gridWidth, dim)
        return concatenated([tokenPart, resized], axis: 1)
    }

    /// - Parameter x: NHWC image batch with 3 channels.
    /// - Returns: the two ResNet taps and the two ViT taps DPT reassembles, in DPT order.
    func callAsFunction(_ x: MLXArray, tapLayers: [Int], recorder: TraceRecorder? = nil) -> (
        layer1: MLXArray, layer2: MLXArray, taps: [MLXArray]
    ) {
        let (stage0, stage1, features) = patchEmbed.backbone(x, recorder: recorder)

        // (B, H/16, W/16, C) -> (B, N, C)
        let projected = patchEmbed.proj(features)
        recorder?.record("patch_proj", projected)
        let gridHeight = projected.dim(1)
        let gridWidth = projected.dim(2)
        let batch = projected.dim(0)
        var h = projected.reshaped(batch, gridHeight * gridWidth, projected.dim(3))

        let cls = broadcast(clsToken, to: [batch, 1, h.dim(2)])
        h = concatenated([cls, h], axis: 1)
        h = h + resizedPositionEmbedding(gridHeight: gridHeight, gridWidth: gridWidth)

        var taps: [MLXArray] = []
        for (index, block) in blocks.enumerated() {
            h = block(h)
            if index == 0 { recorder?.record("block0", h) }
            if tapLayers.contains(index) {
                taps.append(h)
                recorder?.record("tap\(index)", h)
            }
        }

        return (stage0, stage1, taps)
    }
}
