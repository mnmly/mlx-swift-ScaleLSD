// Vanishing-point estimation has no upstream reference to compare against, so it is verified
// against synthetic scenes whose vanishing points are known by construction.

import Foundation
import Testing

@testable import MLXScaleLSD

struct VanishingPointTests {

    /// Build segments that all point at `(px, py)`, spread over a range of positions.
    ///
    /// Each segment is a short chord of the ray from a start point toward the vanishing point,
    /// so the pencil converges by construction.
    private func segments(
        towards px: Float, _ py: Float, count: Int, seed: UInt64 = 1, jitterDegrees: Float = 0
    ) -> [LineSegment] {
        var random = SystemlessRandom(seed: seed)
        return (0 ..< count).map { index in
            let startX = Float(random.unit()) * 512
            let startY = Float(random.unit()) * 512
            var dx = px - startX
            var dy = py - startY
            let length = (dx * dx + dy * dy).squareRoot()
            dx /= length
            dy /= length

            if jitterDegrees != 0 {
                let angle = (Float(random.unit()) * 2 - 1) * jitterDegrees * .pi / 180
                let (c, s) = (cos(angle), sin(angle))
                (dx, dy) = (c * dx - s * dy, s * dx + c * dy)
            }

            let extent: Float = 40 + Float(random.unit()) * 60
            _ = index
            return LineSegment(
                x1: startX, y1: startY,
                x2: startX + dx * extent, y2: startY + dy * extent,
                score: 20, startJunction: 0, endJunction: 1)
        }
    }

    @Test("Recovers a single finite vanishing point")
    func singleVanishingPoint() {
        let truth = (x: Float(900), y: Float(240))
        let found = VanishingPointEstimator.estimate(
            segments: segments(towards: truth.x, truth.y, count: 60),
            options: .init(maximumCount: 1))

        #expect(found.count == 1)
        let point = try! #require(found.first?.imagePoint)
        #expect(abs(point.x - truth.x) < 5)
        #expect(abs(point.y - truth.y) < 5)
    }

    @Test("Separates two vanishing points in one scene")
    func twoVanishingPoints() {
        let a = (x: Float(1400), y: Float(260))
        let b = (x: Float(-800), y: Float(300))
        var scene = segments(towards: a.x, a.y, count: 50, seed: 2)
        scene += segments(towards: b.x, b.y, count: 50, seed: 3)

        let found = VanishingPointEstimator.estimate(
            segments: scene, options: .init(maximumCount: 2))
        #expect(found.count == 2)

        // Each ground-truth point must be matched by one of the results.
        for truth in [a, b] {
            let matched = found.contains { candidate in
                guard let point = candidate.imagePoint else { return false }
                return abs(point.x - truth.x) < 20 && abs(point.y - truth.y) < 20
            }
            #expect(matched, "no vanishing point recovered near \(truth)")
        }
    }

    @Test("Handles parallel lines as a point at infinity")
    func pointAtInfinity() {
        // Perfectly horizontal segments never converge; the vanishing point is at infinity.
        let scene = (0 ..< 40).map { index -> LineSegment in
            let y = Float(index) * 12
            return LineSegment(
                x1: 20, y1: y, x2: 480, y2: y, score: 20, startJunction: 0, endJunction: 1)
        }
        let found = VanishingPointEstimator.estimate(
            segments: scene, options: .init(maximumCount: 1))

        #expect(found.count == 1)
        let point = try! #require(found.first)
        #expect(point.imagePoint == nil, "expected a point at infinity, got \(point)")
        // The direction of the pencil is still well defined, and horizontal.
        let direction = point.direction(at: (x: 250, y: 250))
        #expect(abs(abs(direction.x) - 1) < 1e-3)
        #expect(abs(direction.y) < 1e-3)
    }

    @Test("Tolerates angular noise on the segments")
    func robustToNoise() {
        let truth = (x: Float(1100), y: Float(180))
        let found = VanishingPointEstimator.estimate(
            segments: segments(towards: truth.x, truth.y, count: 80, seed: 5, jitterDegrees: 1.0),
            options: .init(maximumCount: 1, angleThreshold: 3))

        let point = try! #require(found.first?.imagePoint)
        #expect(abs(point.x - truth.x) < 60)
        #expect(abs(point.y - truth.y) < 60)
    }

    @Test("Ignores scenes with too little support")
    func rejectsWeakEvidence() {
        let scene = segments(towards: 900, 240, count: 4)
        let found = VanishingPointEstimator.estimate(
            segments: scene, options: .init(minimumSupport: 8))
        #expect(found.isEmpty)
    }

    @Test("Is deterministic for a fixed seed")
    func deterministic() {
        let scene = segments(towards: 900, 240, count: 60, seed: 7)
        let first = VanishingPointEstimator.estimate(segments: scene)
        let second = VanishingPointEstimator.estimate(segments: scene)
        #expect(first == second)
    }
}

/// Deterministic uniform source for building fixtures, independent of the estimator's own RNG.
private struct SystemlessRandom {
    private var state: UInt64
    init(seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1 }

    mutating func unit() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
}
