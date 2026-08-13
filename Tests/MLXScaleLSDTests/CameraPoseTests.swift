// Verified by round-trip: build a camera with a known focal length and orientation, project
// three orthogonal scene directions through it to get vanishing points, then check the estimator
// recovers what went in.

import Foundation
import Testing
import simd

@testable import MLXScaleLSD

struct CameraPoseTests {

    private let size = (width: 640, height: 480)

    /// Project a scene direction through a camera into a vanishing point.
    private func vanishingPoint(
        direction: SIMD3<Float>, rotation: simd_float3x3, focal: Float, support: Float = 500
    ) -> VanishingPoint? {
        let camera = rotation * direction
        // Behind the camera images to the same point; flip so z > 0.
        let d = camera.z < 0 ? -camera : camera
        guard abs(d.z) > 1e-5 else { return nil }  // at infinity
        let x = Float(size.width) / 2 + focal * d.x / d.z
        let y = Float(size.height) / 2 + focal * d.y / d.z
        return VanishingPoint(x: x, y: y, w: 1, supportingSegments: [], support: support)
    }

    /// Rotation about X then Y then Z, in degrees.
    private func rotation(pitch: Float, yaw: Float, roll: Float) -> simd_float3x3 {
        func radians(_ d: Float) -> Float { d * .pi / 180 }
        let x = simd_float3x3(simd_quatf(angle: radians(pitch), axis: SIMD3(1, 0, 0)))
        let y = simd_float3x3(simd_quatf(angle: radians(yaw), axis: SIMD3(0, 1, 0)))
        let z = simd_float3x3(simd_quatf(angle: radians(roll), axis: SIMD3(0, 0, 1)))
        return z * x * y
    }

    @Test("Recovers a known focal length")
    func recoversFocal() {
        let truth: Float = 600
        // Yaw so both horizontal directions stay finite in the image.
        let r = rotation(pitch: 0, yaw: 30, roll: 0)
        let points = [
            vanishingPoint(direction: SIMD3(1, 0, 0), rotation: r, focal: truth),
            vanishingPoint(direction: SIMD3(0, 0, 1), rotation: r, focal: truth),
        ].compactMap { $0 }
        #expect(points.count == 2)

        let pose = try! #require(
            CameraPoseEstimator.estimate(vanishingPoints: points, imageSize: size))
        #expect(abs(pose.focalLength - truth) < 1)
    }

    @Test("Focal is independent of the rotation used to observe it")
    func focalAcrossRotations() {
        let truth: Float = 720
        for yaw in stride(from: Float(20), through: 70, by: 10) {
            let r = rotation(pitch: 8, yaw: yaw, roll: 0)
            let points = [
                vanishingPoint(direction: SIMD3(1, 0, 0), rotation: r, focal: truth),
                vanishingPoint(direction: SIMD3(0, 0, 1), rotation: r, focal: truth),
            ].compactMap { $0 }
            guard points.count == 2 else { continue }
            let pose = try! #require(
                CameraPoseEstimator.estimate(vanishingPoints: points, imageSize: size))
            #expect(abs(pose.focalLength - truth) < 2, "yaw \(yaw) gave \(pose.focalLength)")
        }
    }

    @Test("Recovers roll")
    func recoversRoll() {
        let focal: Float = 600
        for truth in [Float(-12), -5, 0, 7, 15] {
            let r = rotation(pitch: 0, yaw: 35, roll: truth)
            let points = [
                vanishingPoint(direction: SIMD3(1, 0, 0), rotation: r, focal: focal),
                vanishingPoint(direction: SIMD3(0, 0, 1), rotation: r, focal: focal),
                vanishingPoint(direction: SIMD3(0, 1, 0), rotation: r, focal: focal),
            ].compactMap { $0 }

            let pose = try! #require(
                CameraPoseEstimator.estimate(vanishingPoints: points, imageSize: size))
            // Roll is signed about the optical axis; compare modulo the 180° gauge.
            let error = abs(abs(pose.roll) - abs(truth))
            #expect(error < 1.5, "roll \(truth) recovered as \(pose.roll)")
        }
    }

    @Test("Recovered axes are orthonormal and right-handed")
    func rotationIsValid() {
        let r = rotation(pitch: 5, yaw: 40, roll: 6)
        let points = [
            vanishingPoint(direction: SIMD3(1, 0, 0), rotation: r, focal: 650),
            vanishingPoint(direction: SIMD3(0, 0, 1), rotation: r, focal: 650),
        ].compactMap { $0 }

        let pose = try! #require(
            CameraPoseEstimator.estimate(vanishingPoints: points, imageSize: size))
        let m = pose.rotation
        #expect(abs(simd_determinant(m) - 1) < 1e-3)
        for i in 0 ..< 3 {
            #expect(abs(simd_length(m[i]) - 1) < 1e-3)
            for j in (i + 1) ..< 3 {
                #expect(abs(simd_dot(m[i], m[j])) < 1e-3)
            }
        }
    }

    @Test("An identity rotation homography leaves the image alone")
    func identityHomography() {
        let points = [
            vanishingPoint(
                direction: SIMD3(1, 0, 0), rotation: rotation(pitch: 0, yaw: 30, roll: 0),
                focal: 600),
            vanishingPoint(
                direction: SIMD3(0, 0, 1), rotation: rotation(pitch: 0, yaw: 30, roll: 0),
                focal: 600),
        ].compactMap { $0 }
        let pose = try! #require(
            CameraPoseEstimator.estimate(vanishingPoints: points, imageSize: size))

        let h = pose.rotationHomography(matrix_identity_float3x3)
        let point = SIMD3<Float>(123, 456, 1)
        let mapped = h * point
        #expect(abs(mapped.x / mapped.z - 123) < 1e-2)
        #expect(abs(mapped.y / mapped.z - 456) < 1e-2)
    }

    @Test("Rejects a non-orthogonal pair rather than inventing a focal length")
    func rejectsNonOrthogonal() {
        // Two vanishing points on the same side of the principal point are not orthogonal:
        // (v1-p)·(v2-p) > 0 leaves no real focal length.
        let points = [
            VanishingPoint(x: 900, y: 300, w: 1, supportingSegments: [], support: 500),
            VanishingPoint(x: 1200, y: 320, w: 1, supportingSegments: [], support: 500),
        ]
        #expect(CameraPoseEstimator.estimate(vanishingPoints: points, imageSize: size) == nil)
    }

    @Test("Needs at least two finite vanishing points")
    func rejectsInsufficientInput() {
        let atInfinity = VanishingPoint(x: 1, y: 0, w: 0, supportingSegments: [], support: 500)
        let finite = VanishingPoint(x: 900, y: 240, w: 1, supportingSegments: [], support: 500)
        #expect(CameraPoseEstimator.estimate(vanishingPoints: [finite], imageSize: size) == nil)
        #expect(
            CameraPoseEstimator.estimate(
                vanishingPoints: [finite, atInfinity], imageSize: size) == nil)
    }
}
