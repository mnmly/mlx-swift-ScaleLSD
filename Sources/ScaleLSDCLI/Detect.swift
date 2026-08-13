// `scalelsd detect` — mirrors predictor/predict.py.

import ArgumentParser
import Foundation
import MLXScaleLSD

/// Options shared by every subcommand that runs the network.
struct ModelOptions: ParsableArguments {
    @Option(
        name: [.customShort("m"), .long],
        help: "Directory holding config.json + model.safetensors (see Scripts/convert.py).")
    var model: String

    @Option(name: [.customShort("t"), .long], help: "Minimum segment score to keep.")
    var threshold: Float = 10

    @Option(name: [.customLong("junction-hm"), .customShort("j")],
            help: "Junction heatmap threshold.")
    var junctionThreshold: Float = 0.008

    @Option(name: [.customLong("num-junctions"), .customShort("n")],
            help: "Maximum junctions to detect.")
    var numJunctions: Int = 512

    @Flag(name: .customLong("use-nms"), help: "Apply 3x3 NMS to the junction heatmap.")
    var useNMS = false

    @Flag(
        name: .customLong("use-lsd"),
        help: "Take segment direction from classical LSD instead of the network.")
    var useLSD = false

    @Option(help: "Square resolution the image is resized to before inference.")
    var inputSize: Int = 512

    var scaleLSDOptions: ScaleLSDOptions {
        ScaleLSDOptions(
            inputSize: inputSize,
            decoder: DecoderOptions(
                junctionThreshold: junctionThreshold,
                maximumJunctions: numJunctions,
                useNMS: useNMS),
            useLSD: useLSD)
    }
}

struct Detect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Detect line segments in an image or a folder of images."
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: [.customShort("i"), .long], help: "Image file, or a folder of images.")
    var image: String

    @Option(name: [.customShort("e"), .long], help: "Output format: png or json.")
    var ext: String = "png"

    @Option(help: "Output directory. Defaults to ./temp_output/ScaleLSD.")
    var saveTo: String?

    @Option(help: "White wash over the source image in PNG output, 0...1.")
    var whiteBackground: Double = 0

    @Flag(
        name: [.customLong("vanishing-points"), .customShort("v")],
        help: "Estimate vanishing points and colour segments by the one they support.")
    var vanishingPoints = false

    func run() async throws {
        guard ["png", "json"].contains(ext) else {
            throw ValidationError("--ext must be png or json")
        }

        let inputs = try Self.imageURLs(at: image)
        guard !inputs.isEmpty else { throw ValidationError("no .png/.jpg images at \(image)") }

        let outputDirectory = URL(filePath: saveTo ?? "temp_output/ScaleLSD")
        try FileManager.default.createDirectory(
            at: outputDirectory, withIntermediateDirectories: true)

        let session = try ScaleLSDSession.load(directory: URL(filePath: modelOptions.model))
        let options = modelOptions.scaleLSDOptions

        for url in inputs {
            // Cadence and reporting are frontend concerns; the session does the work.
            let source = try ScaleLSDSession.loadImage(at: url)
            let result = try session.detect(source, options: options)
            let kept = result.segments(minimumScore: modelOptions.threshold)
            // Indices in `supportingSegments` refer to `kept`, so estimate and render use it.
            let vps =
                vanishingPoints
                ? VanishingPointEstimator.estimate(segments: kept) : []

            switch ext {
            case "png":
                let style = WireframeRenderer.Style(whiteOverlay: CGFloat(whiteBackground))
                let annotated = try WireframeRenderer.render(
                    image: source, segments: kept, junctions: result.junctions,
                    vanishingPoints: vps, style: style)
                let destination = outputDirectory
                    .appending(path: url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("png")
                try WireframeRenderer.writePNG(annotated, to: destination)
            default:
                let destination = outputDirectory
                    .appending(path: url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("json")
                try WireframeJSON.write(
                    result, keeping: kept, vanishingPoints: vps, to: destination)
            }

            var summary =
                "\(url.lastPathComponent): \(kept.count) segments, "
                + "\(result.junctions.count) junctions, "
                + String(format: "%.3f s", result.inferenceDuration)
            if vanishingPoints {
                let described = vps.map { vanishing -> String in
                    let where_ = vanishing.imagePoint.map {
                        String(format: "(%.0f, %.0f)", $0.x, $0.y)
                    } ?? "at infinity"
                    return "\(where_)x\(vanishing.supportingSegments.count)"
                }
                summary += "  vps: " + (described.isEmpty ? "none" : described.joined(separator: " "))
                if let pose = CameraPoseEstimator.estimate(
                    vanishingPoints: vps,
                    imageSize: (width: result.imageWidth, height: result.imageHeight))
                {
                    summary += String(
                        format: "\n    camera: focal %.1f px, hFOV %.1f°, roll %.2f°, pitch %.2f°",
                        pose.focalLength,
                        pose.horizontalFieldOfView(width: Float(result.imageWidth)),
                        pose.roll, pose.pitch)
                }
            }
            print(summary)
        }
        print("wrote \(inputs.count) result(s) to \(outputDirectory.path)")
    }

    /// A single image path, or every image directly inside a folder.
    static func imageURLs(at path: String) throws -> [URL] {
        let url = URL(filePath: path)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw ValidationError("no such file or directory: \(path)")
        }
        let extensions = ["png", "jpg", "jpeg"]
        if !isDirectory.boolValue {
            return [url]
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil)
        return entries
            .filter { extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.path < $1.path }
    }
}

/// The wireframe JSON upstream's `--ext json` writes.
enum WireframeJSON {
    static func write(
        _ result: DetectionResult, keeping segments: [LineSegment],
        vanishingPoints: [VanishingPoint] = [], to url: URL
    ) throws {
        let payload: [String: Any] = [
            "width": result.imageWidth,
            "height": result.imageHeight,
            "junctions": result.junctions.map { [$0.x, $0.y] },
            "junction_score": result.junctions.map { $0.score },
            "edges": segments.map { [$0.startJunction, $0.endJunction] },
            "edge_score": segments.map { $0.score },
            "lines": segments.map { [$0.x1, $0.y1, $0.x2, $0.y2] },
            "vanishing_points": vanishingPoints.map { vanishing in
                [
                    // Homogeneous, so a point at infinity is representable.
                    "homogeneous": [vanishing.x, vanishing.y, vanishing.w],
                    "image_point": vanishing.imagePoint.map { [$0.x, $0.y] } as Any,
                    "segments": vanishing.supportingSegments,
                    "support": vanishing.support,
                ] as [String: Any]
            },
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}
