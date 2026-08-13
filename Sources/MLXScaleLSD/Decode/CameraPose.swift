// Camera orientation and focal length from orthogonal vanishing points.
//
// Like `VanishingPoints.swift` this is an addition rather than a port — upstream ScaleLSD has no
// calibration code. It is verified against synthetic cameras with known focal length and
// rotation.
//
// ## What this can and cannot give you
//
// A vanishing point is the image of a 3D *direction*, so a set of them constrains the camera's
// **orientation** and **focal length** — never its position. That is not a limitation of the
// implementation but of the input: recovering translation requires parallax, which a single
// image does not contain.
//
// The useful consequence is that a pure camera rotation maps to a homography, `H = K R K⁻¹`,
// with no depth term. An image can therefore be re-rendered *exactly* as though the camera had
// turned in place — bounded only by the original field of view, since pixels outside the
// original frame were never recorded.

import Foundation
import simd

/// Camera intrinsics and orientation recovered from vanishing points.
public struct CameraPose: Sendable, Equatable {

    /// Focal length in pixels.
    public let focalLength: Float

    /// Principal point in pixels, assumed to be the image centre.
    public let principalPoint: SIMD2<Float>

    /// Rotation from scene axes to camera axes; its columns are the scene's three orthogonal
    /// directions expressed in camera coordinates.
    ///
    /// Determined only up to the gauge freedom inherent in the problem: the labelling of the two
    /// horizontal axes, and the sign of each. It is canonicalised so the vertical scene axis maps
    /// to the column closest to the image's vertical, and so the determinant is `+1`.
    public let rotation: simd_float3x3

    /// Index into the estimate's vanishing points identifying the vertical (up) direction.
    public let verticalAxis: Int

    /// The horizon: the vanishing line of the plane perpendicular to the vertical axis, as
    /// homogeneous line coefficients `(a, b, c)` with `ax + by + c = 0` in pixels.
    public let horizon: SIMD3<Float>

    /// Camera intrinsic matrix.
    public var intrinsics: simd_float3x3 {
        // simd is column-major: each SIMD3 below is a column.
        simd_float3x3(
            SIMD3<Float>(focalLength, 0, 0),
            SIMD3<Float>(0, focalLength, 0),
            SIMD3<Float>(principalPoint.x, principalPoint.y, 1))
    }

    /// Horizontal field of view in degrees, for an image `width` pixels across.
    public func horizontalFieldOfView(width: Float) -> Float {
        2 * atan(width / (2 * focalLength)) * 180 / .pi
    }

    /// Vertical field of view in degrees, for an image `height` pixels tall.
    ///
    /// This is the value to hand a renderer's perspective camera.
    public func verticalFieldOfView(height: Float) -> Float {
        2 * atan(height / (2 * focalLength)) * 180 / .pi
    }

    /// Camera roll in degrees — rotation about the optical axis, i.e. how far off level.
    ///
    /// This is the tilt of the horizon: for `ax + by + c = 0` the line runs along `(b, -a)`.
    /// A line carries no direction, so roll is only defined modulo 180° and is reported in
    /// `(-90, 90]` — a camera rolled 179° is indistinguishable from one rolled -1°.
    public var roll: Float {
        var angle = atan2(-horizon.x, horizon.y) * 180 / .pi
        if angle > 90 { angle -= 180 }
        if angle <= -90 { angle += 180 }
        return angle
    }

    /// Camera pitch in degrees — positive looking up, negative looking down.
    ///
    /// Derived from where the horizon sits relative to the principal point.
    public var pitch: Float {
        let distance =
            (horizon.x * principalPoint.x + horizon.y * principalPoint.y + horizon.z)
            / (horizon.x * horizon.x + horizon.y * horizon.y).squareRoot()
        return atan(distance / focalLength) * 180 / .pi
    }

    /// The homography that re-renders the image as though the camera had rotated by `delta`.
    ///
    /// `H = K · delta · K⁻¹`. Exact for rotation about the optical centre, because no depth
    /// enters the expression.
    public func rotationHomography(_ delta: simd_float3x3) -> simd_float3x3 {
        intrinsics * delta * intrinsics.inverse
    }

    /// Back-project an image point to a unit ray in camera coordinates.
    public func ray(through point: SIMD2<Float>) -> SIMD3<Float> {
        simd_normalize(
            SIMD3<Float>(
                point.x - principalPoint.x, point.y - principalPoint.y, focalLength))
    }
}

/// Recovers ``CameraPose`` from vanishing points.
public enum CameraPoseEstimator {

    /// Plausible focal lengths, as a multiple of image width. Rejects degenerate pairs whose
    /// implied lens is absurd rather than reporting a confident nonsense answer.
    private static let focalRange: ClosedRange<Float> = 0.15 ... 12.0

    /// Estimate intrinsics and orientation from two or more vanishing points.
    ///
    /// Requires at least one pair of *orthogonal, finite* vanishing points: the focal length
    /// comes from the constraint `(v₁ − p) · (v₂ − p) + f² = 0`, which only has a real solution
    /// when the pair is genuinely orthogonal. Pairs that fail it, or that imply an implausible
    /// lens, are skipped — a vanishing point near infinity is very sensitive to noise, so a
    /// three-point set frequently contains only one usable pair.
    ///
    /// - Parameters:
    ///   - vanishingPoints: as returned by ``VanishingPointEstimator/estimate(segments:options:)``.
    ///   - imageSize: pixel dimensions, used for the principal point and the plausibility check.
    /// - Returns: `nil` when no orthogonal pair is usable.
    public static func estimate(
        vanishingPoints: [VanishingPoint], imageSize: (width: Int, height: Int)
    ) -> CameraPose? {
        let principal = SIMD2<Float>(Float(imageSize.width) / 2, Float(imageSize.height) / 2)
        let width = Float(imageSize.width)

        // Finite vanishing points only: a point at infinity gives no focal constraint.
        let finite = vanishingPoints.enumerated().compactMap {
            (index, vanishing) -> (index: Int, point: SIMD2<Float>, support: Float)? in
            guard let image = vanishing.imagePoint else { return nil }
            return (index, SIMD2<Float>(image.x, image.y), vanishing.support)
        }
        guard finite.count >= 2 else { return nil }

        // Best orthogonal pair, preferring the strongest supported one.
        var best: (a: Int, b: Int, focal: Float, support: Float)?
        for i in 0 ..< finite.count {
            for j in (i + 1) ..< finite.count {
                let u = finite[i].point - principal
                let v = finite[j].point - principal
                let squared = -simd_dot(u, v)
                guard squared > 0 else { continue }  // not an orthogonal pair
                let focal = squared.squareRoot()
                guard focalRange.contains(focal / width) else { continue }

                let support = finite[i].support + finite[j].support
                if best == nil || support > best!.support {
                    best = (finite[i].index, finite[j].index, focal, support)
                }
            }
        }
        guard let chosen = best else { return nil }

        let focal = chosen.focal
        func direction(_ index: Int) -> SIMD3<Float> {
            let point = finite.first { $0.index == index }!.point - principal
            return simd_normalize(SIMD3<Float>(point.x, point.y, focal))
        }

        // Two orthogonal scene directions; the third completes a right-handed frame.
        var d0 = direction(chosen.a)
        var d1 = direction(chosen.b)
        // Re-orthogonalise: the measured pair is only approximately perpendicular.
        d1 = simd_normalize(d1 - simd_dot(d1, d0) * d0)
        var d2 = simd_normalize(simd_cross(d0, d1))

        // Label the axis closest to the image's vertical as "up". In camera coordinates the
        // image's y axis is (0, 1, 0), so that is the direction with the largest |y|.
        var axes = [d0, d1, d2]
        let vertical = axes.indices.max { abs(axes[$0].y) < abs(axes[$1].y) } ?? 2

        // Canonicalise signs so "up" points up the image and the frame stays right-handed.
        if axes[vertical].y < 0 { axes[vertical] = -axes[vertical] }
        let horizontals = axes.indices.filter { $0 != vertical }
        d0 = axes[horizontals[0]]
        d1 = axes[horizontals[1]]
        d2 = axes[vertical]
        if simd_determinant(simd_float3x3(d0, d2, d1)) < 0 { d1 = -d1 }

        // The horizon is the vanishing line of the plane normal to "up": every direction
        // perpendicular to the vertical axis images onto it. In homogeneous image coordinates
        // that line is Kᵀ⁻¹ · up, which reduces to the expression below.
        let up = d2
        let horizon = SIMD3<Float>(
            up.x / focal,
            up.y / focal,
            up.z - (up.x * principal.x + up.y * principal.y) / focal)

        return CameraPose(
            focalLength: focal,
            principalPoint: principal,
            rotation: simd_float3x3(d0, d2, d1),
            verticalAxis: vertical,
            horizon: horizon)
    }
}
