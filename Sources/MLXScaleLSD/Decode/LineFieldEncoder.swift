// Encoding a set of line segments into a HAT attraction field.
// PORT FROM: scalelsd/base/csrc/linesegment.cu (encode_kernel)
//            scalelsd/ssl/models/hafm.py (HAFMencoder.lines2hafm)
//
// Upstream ships this as a CUDA extension. The kernel is a per-pixel nearest-segment search,
// which vectorises directly; no custom Metal kernel is needed.
//
// ## Why only the direction channel
//
// `encode_kernel` fills a 6-channel map plus label and `t` maps, and `lines2hafm` derives an
// angle triplet, a distance field and a validity mask from it. The LSD-rectifier path in
// `ScaleLSD.forward_test` then immediately overwrites almost all of it:
//
//     md_pred = stack(md_lsd)                     # from LSD
//     dis_pred = stack(dis_lsd)                   # from LSD
//     md_pred[:, 1:3] = outputs[:, 1:3].sigmoid() # <- network overwrites channels 1 and 2
//     dis_pred = outputs[:, 3:4].sigmoid()        # <- network overwrites the whole distance
//
// So the only value that survives from LSD is `md_pred[:, 0]`, the *direction* to the nearest
// segment. That is what this encoder computes; the discarded outputs are deliberately not
// built. See docs/PARITY.md.

import Foundation
import MLX

public enum LineFieldEncoder {

    /// Direction-to-nearest-segment field, normalised to `0...1`.
    ///
    /// For every grid cell, finds the closest point on the closest segment and encodes the
    /// angle of the vector pointing at it as `atan2(ay, ax) / 2π + 0.5` — matching upstream's
    /// `md_angle_n`.
    ///
    /// - Parameters:
    ///   - segments: `(N, 4)` of `(x1, y1, x2, y2)` in *grid* coordinates (i.e. already divided
    ///     by the network stride).
    ///   - height: grid height.
    ///   - width: grid width.
    /// - Returns: `(1, height, width, 1)` NHWC field. An empty segment list yields zeros, as
    ///   upstream's early return does.
    public static func directionField(
        segments: MLXArray, height: Int, width: Int
    ) -> MLXArray {
        let count = segments.dim(0)
        guard count > 0 else { return MLXArray.zeros([1, height, width, 1]) }

        // (1, N) endpoint components.
        let x1 = segments[0..., 0].reshaped(1, count)
        let y1 = segments[0..., 1].reshaped(1, count)
        let x2 = segments[0..., 2].reshaped(1, count)
        let y2 = segments[0..., 3].reshaped(1, count)
        let dx = x2 - x1
        let dy = y2 - y1
        let squaredLength = dx * dx + dy * dy

        var angles: [MLXArray] = []
        let pixelCount = height * width
        // Bound the (pixels × segments) intermediate; LSD can return a few thousand segments.
        let chunkSize = Swift.max(1, 1 << 22 / Swift.max(count, 1))

        var offset = 0
        while offset < pixelCount {
            let upper = Swift.min(offset + chunkSize, pixelCount)
            let indices = MLXArray(offset ..< upper).asType(.float32)
            let px = (indices % Float(width)).reshaped(upper - offset, 1)
            let py = floor(indices / Float(width)).reshaped(upper - offset, 1)

            // Projection parameter of the pixel onto each segment, clamped to the segment.
            let t = clip(
                ((px - x1) * dx + (py - y1) * dy) / (squaredLength + 1e-6), min: 0, max: 1)
            let ax = x1 + t * dx - px
            let ay = y1 + t * dy - py
            let squaredDistance = ax * ax + ay * ay

            let nearest = argMin(squaredDistance, axis: 1, keepDims: true)
            let nearestX = takeAlong(ax, nearest, axis: 1)
            let nearestY = takeAlong(ay, nearest, axis: 1)

            angles.append(atan2(nearestY, nearestX))
            offset = upper
        }

        let angle = concatenated(angles, axis: 0).reshaped(1, height, width, 1)
        return angle / (2 * Float.pi) + 0.5
    }
}
