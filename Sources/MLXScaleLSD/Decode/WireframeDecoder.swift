// Turning a decoded HAT field into a wireframe: junctions, then junction-snapped segments.
// PORT FROM: scalelsd/ssl/models/detector.py (get_junctions, wireframe_matcher, forward_test)

import Foundation
import MLX

/// A detected junction, in original-image pixel coordinates.
public struct Junction: Sendable, Hashable {
    public let x: Float
    public let y: Float
    /// Junction-heatmap probability.
    public let score: Float
}

/// A detected line segment, in original-image pixel coordinates.
///
/// `score` is the number of HAT-field pixels that voted for this junction pair, so it is a
/// count rather than a probability — upstream's default display threshold is 10.
public struct LineSegment: Sendable, Hashable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float
    public let score: Float

    /// Indices into the result's `junctions` array.
    public let startJunction: Int
    public let endJunction: Int
}

/// Decoder settings, all cheap to change without re-running the network.
public struct DecoderOptions: Sendable, Equatable {
    /// Junction-heatmap threshold (upstream `--junction-hm`).
    public var junctionThreshold: Float
    /// Maximum junctions retained (upstream `--num-junctions`).
    public var maximumJunctions: Int
    /// Apply 3×3 non-maximum suppression to the junction heatmap (upstream `--use_nms`).
    public var useNMS: Bool
    /// Maximum squared distance, in output pixels, from a proposed endpoint to the junction it
    /// snaps to (upstream `j2l_threshold`).
    public var junctionToLineThreshold: Float

    public init(
        junctionThreshold: Float = 0.008,
        maximumJunctions: Int = 512,
        useNMS: Bool = false,
        junctionToLineThreshold: Float = 10
    ) {
        self.junctionThreshold = junctionThreshold
        self.maximumJunctions = maximumJunctions
        self.useNMS = useNMS
        self.junctionToLineThreshold = junctionToLineThreshold
    }

    public static let `default` = DecoderOptions()
}

/// Decodes a HAT field into a wireframe.
public enum WireframeDecoder {

    /// - Parameters:
    ///   - field: the activated network output.
    ///   - imageSize: original image size, which the output is rescaled to.
    ///   - distanceThreshold: the field's distance normalisation.
    ///   - options: junction and matching thresholds.
    /// - Returns: every candidate segment, unfiltered by score. Filtering is the caller's job so
    ///   a UI can re-threshold without re-running the network.
    public static func decode(
        field: HATField,
        imageSize: (width: Int, height: Int),
        distanceThreshold: Float,
        options: DecoderOptions
    ) -> (junctions: [Junction], lines: [LineSegment]) {

        let heatmap = field.suppressedJunctionHeatmap(kernelSize: options.useNMS ? 3 : 1)
        let junctions = extractJunctions(
            heatmap: heatmap, offsets: field.junctionOffsets, options: options)
        guard junctions.count >= 2 else { return ([], []) }

        let proposals = field.decodeLines(distanceThreshold: distanceThreshold)[0]
        let lines = match(
            proposals: proposals, junctions: junctions,
            threshold: options.junctionToLineThreshold)

        // Rescale from output resolution to the original image.
        let scaleX = Float(imageSize.width) / Float(field.width)
        let scaleY = Float(imageSize.height) / Float(field.height)

        let scaledJunctions = junctions.map {
            Junction(x: $0.x * scaleX, y: $0.y * scaleY, score: $0.score)
        }
        let scaledLines = lines.map { edge in
            LineSegment(
                x1: junctions[edge.start].x * scaleX, y1: junctions[edge.start].y * scaleY,
                x2: junctions[edge.end].x * scaleX, y2: junctions[edge.end].y * scaleY,
                score: Float(edge.count),
                startJunction: edge.start, endJunction: edge.end)
        }
        return (scaledJunctions, scaledLines)
    }

    // MARK: - Junctions

    /// Top-k junction peaks with sub-pixel offsets applied, in output-resolution coordinates.
    ///
    /// Runs on the CPU: the selection is a threshold plus a partial sort over `H*W` scalars,
    /// which is cheaper to do directly than to express as MLX gathers.
    private static func extractJunctions(
        heatmap: MLXArray, offsets: MLXArray, options: DecoderOptions
    ) -> [Junction] {
        let width = heatmap.dim(2)
        eval(heatmap, offsets)
        let scores: [Float] = heatmap.asArray(Float.self)
        let offsetValues: [Float] = offsets.asArray(Float.self)

        // Upstream takes exactly as many peaks as clear the threshold, capped at the maximum.
        var candidates: [(index: Int, score: Float)] = []
        candidates.reserveCapacity(1024)
        for index in scores.indices where scores[index] > options.junctionThreshold {
            candidates.append((index, scores[index]))
        }
        candidates.sort { $0.score == $1.score ? $0.index < $1.index : $0.score > $1.score }
        if candidates.count > options.maximumJunctions {
            candidates.removeLast(candidates.count - options.maximumJunctions)
        }

        return candidates.map { candidate in
            let row = candidate.index / width
            let column = candidate.index % width
            // Offsets are interleaved (x, y) on the channel axis.
            let offsetX = offsetValues[candidate.index * 2]
            let offsetY = offsetValues[candidate.index * 2 + 1]
            return Junction(
                x: Float(column) + offsetX + 0.5,
                y: Float(row) + offsetY + 0.5,
                score: candidate.score)
        }
    }

    // MARK: - Matching

    private struct Edge {
        let start: Int
        let end: Int
        let count: Int
    }

    /// Snap every proposed segment to its nearest pair of junctions and tally the votes.
    ///
    /// Upstream scatters the tallies into a `(J, J)` adjacency matrix and then reads back its
    /// upper triangle. Because every kept pair is stored with `start < end` exactly once, that
    /// round-trip is an identity over the unique pairs, so it is skipped here — the resulting
    /// edge order is the same (ascending `start`, then `end`).
    private static func match(
        proposals: MLXArray, junctions: [Junction], threshold: Float
    ) -> [Edge] {
        let junctionCount = junctions.count
        var coordinates = [Float](repeating: 0, count: junctionCount * 2)
        for (index, junction) in junctions.enumerated() {
            coordinates[index * 2] = junction.x
            coordinates[index * 2 + 1] = junction.y
        }
        let junctionArray = MLXArray(coordinates, [junctionCount, 2])

        let (startIndex, startDistance) = nearest(
            points: proposals[0..., ..<2], junctions: junctionArray)
        let (endIndex, endDistance) = nearest(
            points: proposals[0..., 2...], junctions: junctionArray)

        // Tally votes per unordered junction pair.
        var votes: [Int: Int] = [:]
        votes.reserveCapacity(4096)
        for i in startIndex.indices {
            guard startDistance[i] < threshold, endDistance[i] < threshold else { continue }
            let a = Int(startIndex[i])
            let b = Int(endIndex[i])
            guard a != b else { continue }  // both endpoints snapped to one junction
            let key = min(a, b) * junctionCount + max(a, b)
            votes[key, default: 0] += 1
        }

        return votes.keys.sorted().map { key in
            Edge(
                start: key / junctionCount, end: key % junctionCount,
                count: votes[key]!)
        }
    }

    /// Index of, and squared distance to, the nearest junction for each point.
    ///
    /// Chunked over the point axis: the full `(J, H*W)` cost matrix would be ~134 MB at
    /// 512 junctions and a 256×256 field.
    private static func nearest(
        points: MLXArray, junctions: MLXArray
    ) -> (indices: [Int32], distances: [Float]) {
        let total = points.dim(0)
        let chunkSize = 4096
        var indices: [Int32] = []
        var distances: [Float] = []
        indices.reserveCapacity(total)
        distances.reserveCapacity(total)

        var offset = 0
        while offset < total {
            let upper = Swift.min(offset + chunkSize, total)
            let chunk = points[offset ..< upper]
            // (J, chunk, 2) -> (J, chunk)
            let delta = chunk.expandedDimensions(axis: 0) - junctions.expandedDimensions(axis: 1)
            let cost = (delta * delta).sum(axis: -1)

            let bestIndex = argMin(cost, axis: 0)
            let bestDistance = cost.min(axis: 0)
            eval(bestIndex, bestDistance)
            indices.append(contentsOf: bestIndex.asArray(Int32.self))
            distances.append(contentsOf: bestDistance.asArray(Float.self))
            offset = upper
        }
        return (indices, distances)
    }
}
