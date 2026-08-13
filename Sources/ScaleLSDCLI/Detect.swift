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

    @Option(help: "Square resolution the image is resized to before inference.")
    var inputSize: Int = 512

    var scaleLSDOptions: ScaleLSDOptions {
        ScaleLSDOptions(
            inputSize: inputSize,
            decoder: DecoderOptions(
                junctionThreshold: junctionThreshold,
                maximumJunctions: numJunctions,
                useNMS: useNMS))
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

            switch ext {
            case "png":
                let annotated = try WireframeRenderer.render(
                    image: source, segments: kept, junctions: result.junctions,
                    style: .init(whiteOverlay: CGFloat(whiteBackground)))
                let destination = outputDirectory
                    .appending(path: url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("png")
                try WireframeRenderer.writePNG(annotated, to: destination)
            default:
                let destination = outputDirectory
                    .appending(path: url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("json")
                try WireframeJSON.write(result, keeping: kept, to: destination)
            }

            print(
                "\(url.lastPathComponent): \(kept.count) segments, "
                    + "\(result.junctions.count) junctions, "
                    + String(format: "%.3f s", result.inferenceDuration))
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
    static func write(_ result: DetectionResult, keeping segments: [LineSegment], to url: URL)
        throws
    {
        let payload: [String: Any] = [
            "width": result.imageWidth,
            "height": result.imageHeight,
            "junctions": result.junctions.map { [$0.x, $0.y] },
            "junction_score": result.junctions.map { $0.score },
            "edges": segments.map { [$0.startJunction, $0.endJunction] },
            "edge_score": segments.map { $0.score },
            "lines": segments.map { [$0.x1, $0.y1, $0.x2, $0.y2] },
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}
