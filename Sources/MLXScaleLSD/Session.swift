// The one driver both frontends consume.
//
// `scalelsd` (CLI) and Examples/ScaleLSDDemo (SwiftUI) must contain no workload code — only
// their own loop, cadence and presentation. Everything else lives here.

import CoreGraphics
import Foundation
import ImageIO
import LSD
import MLX

/// Settings for a detection run.
public struct ScaleLSDOptions: Sendable, Equatable {
    /// Square resolution the image is resized to before the network. Upstream uses 512.
    public var inputSize: Int
    /// Junction and matching thresholds, applied during decode.
    public var decoder: DecoderOptions
    /// Replace the network's predicted segment *direction* with one derived from classical LSD
    /// (upstream's `--use_lsd` "LSD-rectifier"). Everything else still comes from the network.
    public var useLSD: Bool

    public init(
        inputSize: Int = 512, decoder: DecoderOptions = .default, useLSD: Bool = false
    ) {
        self.inputSize = inputSize
        self.decoder = decoder
        self.useLSD = useLSD
    }

    public static let `default` = ScaleLSDOptions()
}

/// The result of one detection, as plain values.
///
/// Deliberately holds no `MLXArray`, so it is genuinely `Sendable` and safe to hand to the
/// main actor. Every candidate segment is retained; ``segments(minimumScore:)`` re-filters in
/// place so a UI threshold slider never re-runs the network.
public struct DetectionResult: Sendable {
    /// Every detected junction, in original-image pixel coordinates.
    public let junctions: [Junction]
    /// Every candidate segment, unfiltered.
    public let segments: [LineSegment]
    /// Size of the image the coordinates refer to.
    public let imageWidth: Int
    public let imageHeight: Int
    /// Wall-clock seconds spent in the network forward pass.
    public let inferenceDuration: Double

    /// Segments scoring at or above `minimumScore`.
    ///
    /// Pure array filtering — cheap enough to call from a slider's value handler.
    public func segments(minimumScore: Float) -> [LineSegment] {
        segments.filter { $0.score >= minimumScore }
    }
}

/// A retained HAT field, so decode options can change without re-running the network.
///
/// Holds `MLXArray`s, which are reference-backed and not themselves `Sendable`. It is marked
/// `@unchecked Sendable` under the same single-driver contract as ``ScaleLSDSession``: a
/// handle may be moved between tasks, but only one task may read it at a time. That lets a
/// frontend produce it on a detached task, store it, and later re-decode from another task —
/// which is the whole point of separating the field from the decode.
public final class FieldHandle: @unchecked Sendable {
    let field: HATField
    let imageWidth: Int
    let imageHeight: Int
    let inferenceDuration: Double

    init(field: HATField, imageWidth: Int, imageHeight: Int, inferenceDuration: Double) {
        self.field = field
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.inferenceDuration = inferenceDuration
    }
}

/// Loads ScaleLSD once and runs detections against it.
///
/// **Threading contract: one driver at a time.** The type is `@unchecked Sendable` because both
/// frontends respect that — the CLI is single-threaded and the SwiftUI app drives it from a
/// single detached task. Sharing one session between two concurrently-stepping callers is not
/// supported; build a second session instead.
///
/// The session takes decoded pixels, never a path it opens itself, so a sandboxed caller can
/// bake a `CGImage` (``ImageProcessing/bakedCopy(of:)``) while its file grant is valid.
public final class ScaleLSDSession: @unchecked Sendable {

    /// The loaded network's static description.
    public var configuration: ScaleLSDConfiguration { model.configuration }

    private let model: ScaleLSD

    private init(model: ScaleLSD) {
        self.model = model
    }

    /// Load a converted checkpoint directory (`config.json` + `model.safetensors`).
    ///
    /// A sandboxed frontend must call `startAccessingSecurityScopedResource()` on `directory`
    /// first — the session assumes no ambient file access.
    public static func load(directory: URL, dtype: DType = .float32) throws -> ScaleLSDSession {
        ScaleLSDSession(model: try ScaleLSD.load(directory: directory, dtype: dtype))
    }

    // MARK: - The expensive half

    /// Run the network and retain its HAT field.
    ///
    /// - Parameters:
    ///   - image: fully decoded pixels. See ``ImageProcessing/bakedCopy(of:)``.
    ///   - options: input resolution and decoder thresholds; only the resolution affects this
    ///     call, the thresholds are carried through for ``detect(_:options:)``.
    public func analyze(_ image: CGImage, options: ScaleLSDOptions = .default) throws
        -> FieldHandle
    {
        let input = try ImageProcessing.networkInput(from: image, size: options.inputSize)

        // The rectifier runs on the same tensor the network sees, quantised the way upstream
        // does it (`np.array(x * 255, dtype=np.uint8)` — a truncation, not a round).
        let direction: MLXArray? =
            options.useLSD
            ? lsdDirectionField(for: input, stride: configuration.stride) : nil

        let start = DispatchTime.now().uptimeNanoseconds
        let output = model(input)
        // Materialise inside the session so a frontend running off the main actor cannot
        // forget to, and so the reported duration is real rather than the cost of building
        // a lazy graph.
        eval(output)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

        return FieldHandle(
            field: HATField(rawOutput: output, directionOverride: direction),
            imageWidth: image.width, imageHeight: image.height,
            inferenceDuration: elapsed)
    }

    /// Detect segments with classical LSD and encode them as a direction field.
    ///
    /// Only the direction survives into the network's output — upstream overwrites the rest of
    /// the LSD field with the network's own predictions — so nothing else is computed here.
    /// See ``LineFieldEncoder``.
    private func lsdDirectionField(for input: MLXArray, stride: Int) -> MLXArray? {
        let size = input.dim(1)
        eval(input)
        let scaled = input.asArray(Float.self).map { value -> UInt8 in
            UInt8(Swift.max(0, Swift.min(255, Int(value * 255))))  // truncate, as upstream does
        }
        let segments = LSDDetector.detect(grayscale: scaled, width: size, height: input.dim(2))
        guard !segments.isEmpty else { return nil }

        // LSD reports Double coordinates; MLX has no float64 on the GPU.
        let flat = segments.flatMap {
            [Float($0.x1), Float($0.y1), Float($0.x2), Float($0.y2)]
        }
        let grid = MLXArray(flat, [segments.count, 4]) / Float(stride)
        return LineFieldEncoder.directionField(
            segments: grid, height: size / stride, width: input.dim(2) / stride)
    }

    // MARK: - The cheap half

    /// Decode a retained field into a wireframe.
    ///
    /// Re-running this with different ``DecoderOptions`` costs a decode, not a forward pass —
    /// this is what a junction-threshold slider should call.
    public func decode(_ handle: FieldHandle, options: DecoderOptions = .default)
        -> DetectionResult
    {
        let (junctions, segments) = WireframeDecoder.decode(
            field: handle.field,
            imageSize: (width: handle.imageWidth, height: handle.imageHeight),
            distanceThreshold: configuration.distanceThreshold,
            options: options)

        return DetectionResult(
            junctions: junctions, segments: segments,
            imageWidth: handle.imageWidth, imageHeight: handle.imageHeight,
            inferenceDuration: handle.inferenceDuration)
    }

    /// Convenience: analyse and decode in one call.
    public func detect(_ image: CGImage, options: ScaleLSDOptions = .default) throws
        -> DetectionResult
    {
        decode(try analyze(image, options: options), options: options.decoder)
    }

    /// Convenience for callers with ambient file access (i.e. the CLI, not the sandboxed app).
    public func detect(contentsOf url: URL, options: ScaleLSDOptions = .default) throws
        -> DetectionResult
    {
        try detect(try Self.loadImage(at: url), options: options)
    }

    /// Decode an image file to materialised pixels.
    ///
    /// Only safe where the caller has ambient file access; a sandboxed frontend should open the
    /// file itself and bake the result.
    public static func loadImage(at url: URL) throws -> CGImage {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ImageProcessingError.cannotDecode }
        return try ImageProcessing.bakedCopy(of: image)
    }
}
