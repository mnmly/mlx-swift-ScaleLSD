// Interpreting the 9-channel network output as a HAT (Holistic Attraction) field.
// PORT FROM: scalelsd/ssl/models/detector.py (hafm_decoding, non_maximum_suppression)

import Foundation
import MLX
import MLXNN

/// The activated HAT field: every output pixel proposes one line segment plus a junction vote.
///
/// Channel layout of the raw network output (`(B, H, W, 9)`):
/// `0..<3` angle field, `3` distance, `4` distance residual, `5..<7` junction logits,
/// `7..<9` junction sub-pixel offsets.
public struct HATField {

    /// `(B, H, W, 3)` — sigmoid of the angle channels.
    public let angles: MLXArray
    /// `(B, H, W, 1)` — sigmoid of the distance channel.
    public let distance: MLXArray
    /// `(B, H, W, 1)` — softmax junction probability (positive class).
    public let junctionHeatmap: MLXArray
    /// `(B, H, W, 2)` — junction sub-pixel offsets in `-0.5...0.5`.
    public let junctionOffsets: MLXArray

    public var height: Int { angles.dim(1) }
    public var width: Int { angles.dim(2) }

    /// Split and activate a raw `(B, H, W, 9)` network output.
    ///
    /// - Parameters:
    ///   - output: the raw network output.
    ///   - directionOverride: an externally computed direction field, `(B, H, W, 1)` normalised
    ///     to `0...1`, replacing the network's own angle channel 0. This is the LSD-rectifier
    ///     path: upstream substitutes the LSD field's direction and keeps everything else from
    ///     the network. See ``LineFieldEncoder``.
    public init(rawOutput output: MLXArray, directionOverride: MLXArray? = nil) {
        if let directionOverride {
            self.angles = concatenated(
                [directionOverride, sigmoid(output[.ellipsis, 1 ..< 3])], axis: -1)
        } else {
            self.angles = sigmoid(output[.ellipsis, ..<3])
        }
        self.distance = sigmoid(output[.ellipsis, 3 ..< 4])
        // Upstream takes softmax over the 2 logits and keeps the positive class.
        self.junctionHeatmap = softmax(output[.ellipsis, 5 ..< 7], axis: -1)[.ellipsis, 1 ..< 2]
        self.junctionOffsets = sigmoid(output[.ellipsis, 7 ..< 9]) - 0.5
    }

    /// Decode every pixel's proposed segment into absolute endpoint coordinates.
    ///
    /// Each pixel stores a direction (`angles[0]`), two opening angles (`angles[1..<3]`) and a
    /// normalised distance to the segment. Rotating the two opening angles by the direction and
    /// scaling by the distance recovers the segment's endpoints relative to that pixel.
    ///
    /// - Parameter distanceThreshold: the field's distance normalisation (upstream `scale`).
    /// - Returns: `(B, H*W, 4)` of `(x1, y1, x2, y2)` in output-resolution pixels.
    public func decodeLines(distanceThreshold: Float) -> MLXArray {
        let h = height
        let w = width

        // Pixel coordinate grids, broadcast over the batch.
        let xs = MLXArray(0 ..< w).asType(.float32).reshaped(1, 1, w, 1)
        let ys = MLXArray(0 ..< h).asType(.float32).reshaped(1, h, 1, 1)

        let field = clip(distance, min: 0, max: 1) * distanceThreshold

        let direction = (angles[.ellipsis, ..<1] - 0.5) * (2 * Float.pi)
        let startAngle = angles[.ellipsis, 1 ..< 2] * (Float.pi / 2)
        let endAngle = -angles[.ellipsis, 2 ..< 3] * (Float.pi / 2)

        let cosD = cos(direction)
        let sinD = sin(direction)
        let tanStart = tan(startAngle)
        let tanEnd = tan(endAngle)

        // Rotate the opening directions into image space and walk `field` pixels along them.
        let x1 = clip((cosD - sinD * tanStart) * field + xs, min: 0, max: Float(w - 1))
        let y1 = clip((sinD + cosD * tanStart) * field + ys, min: 0, max: Float(h - 1))
        let x2 = clip((cosD - sinD * tanEnd) * field + xs, min: 0, max: Float(w - 1))
        let y2 = clip((sinD + cosD * tanEnd) * field + ys, min: 0, max: Float(h - 1))

        let lines = concatenated([x1, y1, x2, y2], axis: -1)
        return lines.reshaped(lines.dim(0), h * w, 4)
    }

    /// Suppress non-maximal junction responses within a `kernelSize` window.
    ///
    /// A `kernelSize` of 1 is the identity, which is upstream's behaviour when NMS is disabled.
    ///
    /// The dilation is computed as a maximum over shifted views rather than with
    /// `MLXNN.MaxPool2d`: that layer builds its pad widths as `[0, 0] + padding + [0, 0]`, which
    /// is six entries for a 4-D NHWC input, and its strided-window shape math then produces a
    /// differently-shaped result that fails to broadcast here.
    public func suppressedJunctionHeatmap(kernelSize: Int) -> MLXArray {
        guard kernelSize > 1 else { return junctionHeatmap }
        let radius = kernelSize / 2
        let h = height
        let w = width

        // -infinity is the identity for a maximum, so the border behaves as PyTorch's
        // `max_pool2d(..., padding=)` does.
        let widths: [IntOrPair] = [
            .init(0), .init(radius), .init(radius), .init(0),
        ]
        let padded = MLX.padded(
            junctionHeatmap, widths: widths, mode: .constant,
            value: MLXArray(-Float.infinity).asType(junctionHeatmap.dtype))

        var dilated = padded[0..., 0 ..< h, 0 ..< w, 0...]
        for dy in 0 ..< kernelSize {
            for dx in 0 ..< kernelSize where !(dy == 0 && dx == 0) {
                dilated = maximum(dilated, padded[0..., dy ..< (dy + h), dx ..< (dx + w), 0...])
            }
        }
        return junctionHeatmap * (junctionHeatmap .== dilated)
    }
}
