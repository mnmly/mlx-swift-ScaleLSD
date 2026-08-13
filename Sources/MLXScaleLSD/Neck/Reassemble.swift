// DPT "reassemble": turn ViT token sequences back into convolutional feature maps.
// PORT FROM: scalelsd/ssl/backbones/dpt/vit.py (_make_vit_b_rn50_backbone act_postprocess3/4)
//
// The hybrid backbone only reassembles the two ViT taps — `act_postprocess1/2` are
// `nn.Identity` because levels 1 and 2 come straight from the ResNet stages.
//
// Upstream stores these as `nn.Sequential`, so the checkpoint keys are positional
// (`act_postprocess4.0.project.0.weight`, `.3.weight`, `.4.weight`). Scripts/convert.py
// flattens them into the named submodules used here.

import Foundation
import MLX
import MLXNN

/// The `project` readout: concatenate the class token onto every patch token and mix.
///
/// Upstream's `ProjectReadout` with `use_readout="project"`; the trailing GELU is part of the
/// same `nn.Sequential`.
final class ProjectReadout: Module {

    let project: Linear

    init(dimensions: Int) {
        self.project = Linear(dimensions * 2, dimensions, bias: true)
        super.init()
    }

    /// - Parameter x: `(B, 1 + N, C)` token sequence including the class token.
    /// - Returns: `(B, N, C)` with the class token broadcast-concatenated and projected away.
    func callAsFunction(_ x: MLXArray, startIndex: Int = 1) -> MLXArray {
        let patches = x[0..., startIndex...]
        let readout = broadcast(x[0..., ..<1], to: patches.shape)
        return gelu(project(concatenated([patches, readout], axis: -1)))
    }
}

/// One DPT reassemble level: readout → token grid → 1×1 projection → optional resampling.
final class Reassemble: Module {

    let readout: ProjectReadout
    /// 1×1 channel projection to this level's DPT width.
    let proj: Conv2d
    /// Level 4 only: a 3×3 stride-2 convolution that halves the resolution to stride 32.
    let resize: Conv2d?

    init(dimensions: Int, outputChannels: Int, downsample: Bool) {
        self.readout = ProjectReadout(dimensions: dimensions)
        self.proj = Conv2d(
            inputChannels: dimensions, outputChannels: outputChannels, kernelSize: .init(1),
            bias: true)
        self.resize =
            downsample
            ? Conv2d(
                inputChannels: outputChannels, outputChannels: outputChannels,
                kernelSize: .init(3), stride: .init(2), padding: .init(1), bias: true)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        let tokens = readout(x)
        // (B, N, C) -> NHWC grid. Upstream does Transpose(1,2) + Unflatten(2, [h, w]) on an
        // NCHW layout, which is the same element order as this reshape in NHWC.
        var h = tokens.reshaped(tokens.dim(0), gridHeight, gridWidth, tokens.dim(2))
        h = proj(h)
        if let resize { h = resize(h) }
        return h
    }
}
