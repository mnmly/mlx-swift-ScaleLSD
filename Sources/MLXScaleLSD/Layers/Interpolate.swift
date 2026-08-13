// Bilinear resampling helpers matching `torch.nn.functional.interpolate`.

import Foundation
import MLX

/// Bilinear resize of an NHWC tensor to an exact target size, `align_corners=False`.
///
/// `MLXNN.Upsample` derives its output size as `Int(scale * dimension)`, which truncates:
/// resizing a 24×24 position grid to 32×32 would ask for a scale of `32/24` and land on 31.
/// Position-embedding resampling needs the target size to be exact, so the sampling grid is
/// built directly here.
///
/// The index mapping is PyTorch's `area_pixel_compute_source_index` for `align_corners=False`:
/// `src = (dst + 0.5) * (in / out) - 0.5`, clamped at zero.
func bilinearResize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
    var h = x
    if h.dim(1) != height { h = resampleAxis(h, axis: 1, outSize: height) }
    if h.dim(2) != width { h = resampleAxis(h, axis: 2, outSize: width) }
    return h
}

private func resampleAxis(_ x: MLXArray, axis: Int, outSize: Int) -> MLXArray {
    let inSize = x.dim(axis)
    let ratio = Float(inSize) / Float(outSize)

    // Source coordinate of each destination sample, clamped into range.
    var position = (MLXArray(0 ..< outSize).asType(.float32) + 0.5) * ratio - 0.5
    position = maximum(position, MLXArray(Float(0)))

    let low = floor(position)
    let weight = position - low
    let lowIndex = low.asType(.int32)
    let highIndex = minimum(lowIndex + 1, MLXArray(Int32(inSize - 1)))

    // Broadcast the interpolation weight against the sampled axis.
    var weightShape = Array(repeating: 1, count: x.ndim)
    weightShape[axis] = outSize
    let w = weight.reshaped(weightShape).asType(x.dtype)

    let lowSample = take(x, lowIndex, axis: axis)
    let highSample = take(x, highIndex, axis: axis)
    return lowSample * (1 - w) + highSample * w
}
