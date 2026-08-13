// Vanishing-point estimation from detected line segments.
//
// NOT a port: upstream ScaleLSD contains no VP code. Its project page shows VP results produced
// by Progressive-X, a separate C++ library. This is an independent implementation of the
// standard approach — sequential RANSAC over segment pairs in homogeneous coordinates, with a
// least-squares refit — verified against synthetic scenes with known vanishing points rather
// than against a reference implementation.
//
// Everything here works on `[LineSegment]`, so it composes with any detection result and needs
// no camera calibration: a vanishing point is just the common intersection of a pencil of image
// lines, which is well defined in homogeneous coordinates even when the lines are parallel and
// the point lies at infinity.

import Foundation

/// A direction in the scene that a group of image lines converge on.
public struct VanishingPoint: Sendable, Hashable {

    /// Homogeneous image coordinates `(x, y, w)`, unit-norm.
    ///
    /// `w` near zero means the supporting lines are parallel in the image and the point lies at
    /// infinity — common for a horizontal facade viewed head-on. Use ``imagePoint`` to get a
    /// finite location when one exists.
    public let x: Float
    public let y: Float
    public let w: Float

    /// Indices into the segment array that voted for this point.
    public let supportingSegments: [Int]

    /// Total length of the supporting segments, in pixels.
    ///
    /// A better measure of dominance than raw count: one long facade edge is stronger evidence
    /// than several short noisy ones.
    public let support: Float

    /// The finite image-plane location, or `nil` when the point is at (or near) infinity.
    public var imagePoint: (x: Float, y: Float)? {
        guard abs(w) > 1e-6 else { return nil }
        return (x / w, y / w)
    }

    /// Unit direction of the pencil of lines converging here, as seen at `point`.
    ///
    /// Defined whether or not the vanishing point is finite, which is why the overlay draws with
    /// this rather than with ``imagePoint``.
    public func direction(at point: (x: Float, y: Float)) -> (x: Float, y: Float) {
        var dx = x - w * point.x
        var dy = y - w * point.y
        let length = (dx * dx + dy * dy).squareRoot()
        guard length > 1e-12 else { return (0, 0) }
        dx /= length
        dy /= length
        return (dx, dy)
    }
}

/// Tuning for ``VanishingPointEstimator``.
public struct VanishingPointOptions: Sendable, Equatable {

    /// Maximum number of vanishing points to extract.
    ///
    /// Three suits a Manhattan-world scene (two horizontal directions plus vertical).
    public var maximumCount: Int

    /// How far a segment's direction may deviate from pointing at the vanishing point, in
    /// degrees, to count as support.
    public var angleThreshold: Float

    /// Minimum number of supporting segments for a vanishing point to be reported.
    public var minimumSupport: Int

    /// RANSAC sampling budget per vanishing point.
    public var iterations: Int

    /// Segments shorter than this are ignored — short segments have unreliable orientation.
    public var minimumSegmentLength: Float

    /// Seed for the sampler.
    ///
    /// Estimation is randomised, so a fixed seed is what keeps results reproducible between
    /// runs and stops a UI overlay flickering when an unrelated control changes.
    public var seed: UInt64

    public init(
        maximumCount: Int = 3,
        angleThreshold: Float = 2,
        minimumSupport: Int = 8,
        iterations: Int = 2000,
        minimumSegmentLength: Float = 10,
        seed: UInt64 = 0x5CA1_E15D
    ) {
        self.maximumCount = maximumCount
        self.angleThreshold = angleThreshold
        self.minimumSupport = minimumSupport
        self.iterations = iterations
        self.minimumSegmentLength = minimumSegmentLength
        self.seed = seed
    }

    public static let `default` = VanishingPointOptions()
}

/// Finds the dominant vanishing points among a set of line segments.
public enum VanishingPointEstimator {

    /// Extract up to ``VanishingPointOptions/maximumCount`` vanishing points.
    ///
    /// Uses sequential RANSAC: repeatedly find the best-supported candidate, refit it to its
    /// inliers, then remove those segments and search again. Returned points are ordered by
    /// decreasing support.
    ///
    /// - Parameters:
    ///   - segments: detected segments, in any consistent coordinate frame.
    ///   - options: thresholds and sampling budget.
    public static func estimate(
        segments: [LineSegment], options: VanishingPointOptions = .default
    ) -> [VanishingPoint] {
        // Work with indices into the caller's array so `supportingSegments` stays meaningful.
        let usable = segments.indices.filter { index in
            length(of: segments[index]) >= options.minimumSegmentLength
        }
        guard usable.count >= 2 else { return [] }

        let lines = segments.map(homogeneousLine)
        let cosThreshold = cos(options.angleThreshold * .pi / 180)

        var remaining = usable
        var found: [VanishingPoint] = []
        var random = SplitMix64(seed: options.seed)

        while found.count < options.maximumCount, remaining.count >= 2 {
            guard
                let candidate = bestCandidate(
                    among: remaining, segments: segments, lines: lines,
                    cosThreshold: cosThreshold, iterations: options.iterations,
                    random: &random)
            else { break }

            // Refit to the consensus set, then re-collect: the refined point usually gains a
            // few segments the two-sample estimate missed.
            var vector = candidate
            for _ in 0 ..< 2 {
                let inliers = consensus(
                    of: vector, among: remaining, segments: segments, cosThreshold: cosThreshold)
                guard inliers.count >= 2 else { break }
                vector = refit(inliers: inliers, lines: lines, segments: segments)
            }

            let inliers = consensus(
                of: vector, among: remaining, segments: segments, cosThreshold: cosThreshold)
            guard inliers.count >= options.minimumSupport else { break }

            found.append(
                VanishingPoint(
                    x: vector.0, y: vector.1, w: vector.2,
                    supportingSegments: inliers.sorted(),
                    support: inliers.reduce(0) { $0 + length(of: segments[$1]) }))

            let claimed = Set(inliers)
            remaining.removeAll { claimed.contains($0) }
        }

        return found.sorted { $0.support > $1.support }
    }

    // MARK: - Geometry

    private typealias Vector3 = (Float, Float, Float)

    private static func length(of segment: LineSegment) -> Float {
        let dx = segment.x2 - segment.x1
        let dy = segment.y2 - segment.y1
        return (dx * dx + dy * dy).squareRoot()
    }

    /// The homogeneous line through a segment's endpoints, `p1 × p2`.
    private static func homogeneousLine(_ segment: LineSegment) -> Vector3 {
        cross(
            (segment.x1, segment.y1, 1),
            (segment.x2, segment.y2, 1))
    }

    private static func cross(_ a: Vector3, _ b: Vector3) -> Vector3 {
        (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
    }

    private static func normalized(_ v: Vector3) -> Vector3? {
        let n = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).squareRoot()
        guard n > 1e-12 else { return nil }
        return (v.0 / n, v.1 / n, v.2 / n)
    }

    /// Whether a segment points at `vanishing` within the angular threshold.
    ///
    /// Compares the segment's own direction against the direction from its midpoint to the
    /// vanishing point. This is the scale-free test — using distance from the point to the
    /// segment's infinite line instead would penalise far-away vanishing points unfairly.
    private static func supports(
        _ vanishing: Vector3, _ segment: LineSegment, cosThreshold: Float
    ) -> Bool {
        let midX = (segment.x1 + segment.x2) / 2
        let midY = (segment.y1 + segment.y2) / 2

        var towardX = vanishing.0 - vanishing.2 * midX
        var towardY = vanishing.1 - vanishing.2 * midY
        let towardLength = (towardX * towardX + towardY * towardY).squareRoot()
        guard towardLength > 1e-9 else { return false }
        towardX /= towardLength
        towardY /= towardLength

        var dirX = segment.x2 - segment.x1
        var dirY = segment.y2 - segment.y1
        let dirLength = (dirX * dirX + dirY * dirY).squareRoot()
        guard dirLength > 1e-9 else { return false }
        dirX /= dirLength
        dirY /= dirLength

        // Orientation is undirected, so compare magnitudes.
        return abs(towardX * dirX + towardY * dirY) >= cosThreshold
    }

    private static func consensus(
        of vanishing: Vector3, among indices: [Int], segments: [LineSegment],
        cosThreshold: Float
    ) -> [Int] {
        indices.filter { supports(vanishing, segments[$0], cosThreshold: cosThreshold) }
    }

    private static func bestCandidate(
        among indices: [Int], segments: [LineSegment], lines: [Vector3],
        cosThreshold: Float, iterations: Int, random: inout SplitMix64
    ) -> Vector3? {
        var best: Vector3?
        var bestScore: Float = 0

        for _ in 0 ..< iterations {
            let a = indices[Int(random.next(upperBound: UInt64(indices.count)))]
            let b = indices[Int(random.next(upperBound: UInt64(indices.count)))]
            guard a != b else { continue }
            guard let candidate = normalized(cross(lines[a], lines[b])) else { continue }

            // Score by supported length rather than count, for the same reason `support` is.
            var score: Float = 0
            for index in indices
            where supports(candidate, segments[index], cosThreshold: cosThreshold) {
                score += length(of: segments[index])
            }
            if score > bestScore {
                bestScore = score
                best = candidate
            }
        }
        return best
    }

    /// Least-squares vanishing point: the unit vector minimising `Σ (lᵢ · v)²`.
    ///
    /// That is the eigenvector of `Σ lᵢlᵢᵀ` with the smallest eigenvalue. The matrix is a
    /// symmetric 3×3, so a few Jacobi sweeps solve it exactly without pulling in LAPACK.
    private static func refit(
        inliers: [Int], lines: [Vector3], segments: [LineSegment]
    ) -> Vector3 {
        var m = [[Float]](repeating: [0, 0, 0], count: 3)
        for index in inliers {
            guard let line = normalized(lines[index]) else { continue }
            // Weight by length: a long segment localises its line far better than a short one.
            let weight = length(of: segments[index])
            let l = [line.0, line.1, line.2]
            for row in 0 ..< 3 {
                for column in 0 ..< 3 {
                    m[row][column] += weight * l[row] * l[column]
                }
            }
        }
        let (vectors, values) = symmetricEigen3(m)
        let smallest = values.indices.min { values[$0] < values[$1] } ?? 0
        let v = (vectors[0][smallest], vectors[1][smallest], vectors[2][smallest])
        return normalized(v) ?? v
    }

    /// Cyclic Jacobi eigendecomposition of a symmetric 3×3 matrix.
    ///
    /// - Returns: eigenvectors as columns, and the corresponding eigenvalues.
    private static func symmetricEigen3(_ input: [[Float]]) -> (
        vectors: [[Float]], values: [Float]
    ) {
        var a = input
        var v: [[Float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for _ in 0 ..< 12 {
            // Zero the largest off-diagonal entry.
            var p = 0
            var q = 1
            var largest = abs(a[0][1])
            for (i, j) in [(0, 2), (1, 2)] where abs(a[i][j]) > largest {
                largest = abs(a[i][j])
                p = i
                q = j
            }
            if largest < 1e-12 { break }

            let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
            let sign: Float = theta >= 0 ? 1 : -1
            let t = sign / (abs(theta) + (theta * theta + 1).squareRoot())
            let c = 1 / (t * t + 1).squareRoot()
            let s = t * c

            for k in 0 ..< 3 {
                let akp = a[k][p]
                let akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            }
            for k in 0 ..< 3 {
                let apk = a[p][k]
                let aqk = a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            }
            for k in 0 ..< 3 {
                let vkp = v[k][p]
                let vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }
        return (v, [a[0][0], a[1][1], a[2][2]])
    }
}

/// Small deterministic PRNG, so estimation is reproducible across runs.
private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func next(upperBound: UInt64) -> UInt64 {
        upperBound == 0 ? 0 : next() % upperBound
    }
}
