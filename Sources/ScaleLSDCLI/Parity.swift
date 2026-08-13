// `scalelsd parity` — bisect the Swift port against PyTorch fixtures.
//
// Fixtures come from Scripts/make_fixtures.py. Stages are compared in execution order, so the
// first row whose relative error jumps is the first broken layer.

import ArgumentParser
import Foundation
import MLX
import MLXScaleLSD

struct Parity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Compare stage activations against PyTorch reference fixtures."
    )

    @Option(name: [.short, .long], help: "Directory holding config.json + model.safetensors.")
    var model: String

    @Option(name: [.short, .long], help: "Path to fixtures.safetensors.")
    var fixtures: String

    @Option(help: "Relative-error threshold for a stage to count as passing.")
    var tolerance: Float = 2e-4

    @Flag(help: "Also report interior-only error, isolating padding bugs from precision.")
    var detail = false

    @Option(
        name: [.customShort("i"), .long],
        help: "Source image, to also check preprocessing against the reference tensor.")
    var image: String?

    func run() async throws {
        // MLX runs fp32 matmuls at TF32 by default (~1e-3 relative); parity needs it off.
        if ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0" {
            print("warning: run with MLX_ENABLE_TF32=0 for meaningful fp32 tolerances")
        }

        let reference = try loadArrays(url: URL(filePath: fixtures))
        guard let input = reference["input"] else {
            throw ValidationError("fixtures are missing the `input` tensor")
        }

        let network = try ScaleLSD.load(directory: URL(filePath: model))
        let (_, trace) = network.traced(input)

        // Execution order, finest-grained first, so the first failure localises the bug.
        let stages = [
            "stem", "stage0", "stage1", "stage2", "patch_proj", "block0", "tap8", "tap11",
            "layer3", "layer4", "layer1_rn", "layer2_rn", "layer3_rn", "layer4_rn",
            "path4", "path3", "path2", "path1", "output",
        ]

        func pad(_ text: String, _ width: Int) -> String {
            text.count >= width
                ? text : text + String(repeating: " ", count: width - text.count)
        }
        func scientific(_ value: Float) -> String {
            pad(String(format: "%.3e", value), 12)
        }

        print(pad("stage", 12) + pad("shape", 24) + pad("max rel", 13) + pad("mean abs", 13))
        var failures = 0
        for name in stages {
            guard let expected = reference[name] else { continue }
            guard let actual = trace[name] else {
                print("\(name): MISSING from Swift trace")
                failures += 1
                continue
            }
            guard actual.shape == expected.shape else {
                print("\(name): shape \(actual.shape) != reference \(expected.shape)")
                failures += 1
                continue
            }

            let difference = abs(actual - expected)
            let scale = maximum(abs(expected).max(), MLXArray(Float(1e-12)))
            let relative = (difference.max() / scale).item(Float.self)
            let meanAbs = difference.mean().item(Float.self)

            let ok = relative <= tolerance
            if !ok { failures += 1 }
            var line =
                pad(name, 12) + pad("\(expected.shape)", 24)
                + scientific(relative) + " " + scientific(meanAbs)
                + " " + (ok ? "ok" : "FAIL")

            // Split border from interior: a padding bug concentrates error on the edges,
            // while reduced-precision compute spreads it uniformly.
            if detail, expected.ndim == 4, expected.dim(1) > 8, expected.dim(2) > 8 {
                let inset = 4
                let h = expected.dim(1)
                let w = expected.dim(2)
                let interiorDiff = abs(
                    actual[0..., inset ..< (h - inset), inset ..< (w - inset), 0...]
                        - expected[0..., inset ..< (h - inset), inset ..< (w - inset), 0...])
                let interiorRel = (interiorDiff.max() / scale).item(Float.self)
                line += "   interior=" + String(format: "%.3e", interiorRel)
            }
            print(line)
        }

        // ---- preprocessing --------------------------------------------------
        // The fixtures store the *already preprocessed* tensor, so the stage comparison below
        // never exercises grayscale conversion or the resize. Check them explicitly when a
        // source image is supplied, otherwise a preprocessing regression is invisible here.
        if let image {
            let decoded = try ScaleLSDSession.loadImage(at: URL(filePath: image))
            let produced = try ImageProcessing.networkInput(from: decoded, size: input.dim(1))
            let difference = abs(produced - input)
            let relative =
                (difference.max() / maximum(abs(input).max(), MLXArray(Float(1e-12))))
                .item(Float.self)
            // cv2 resizes 8-bit with fixed-point arithmetic, so exact equality is not expected;
            // this is a regression guard on the order of a quantisation step.
            let ok = relative <= 0.02
            if !ok { failures += 1 }
            print(
                pad("preprocess", 12) + pad("\(input.shape)", 24)
                    + scientific(relative) + " " + scientific(difference.mean().item(Float.self))
                    + " " + (ok ? "ok" : "FAIL"))
        }

        // ---- LSD direction field (encodels port) ----------------------------
        if let lsdLines = reference["lsd_lines"], let expectedField = reference["lsd_direction"] {
            let actual = LineFieldEncoder.directionField(
                segments: lsdLines / 2,  // stride
                height: expectedField.dim(1), width: expectedField.dim(2))
            let difference = abs(actual - expectedField)
            let relative =
                (difference.max() / maximum(abs(expectedField).max(), MLXArray(Float(1e-12))))
                .item(Float.self)
            let ok = relative <= tolerance
            if !ok { failures += 1 }
            print(
                pad("lsd_dir", 12) + pad("\(expectedField.shape)", 24)
                    + scientific(relative) + " " + scientific(difference.mean().item(Float.self))
                    + " " + (ok ? "ok" : "FAIL") + "  (\(lsdLines.dim(0)) segments)")
        }

        // ---- end-to-end detections ------------------------------------------
        if let sizeArray = reference["image_size"],
            let referenceLines = reference["lines_pred"],
            let referenceScores = reference["lines_score"],
            let referenceJunctions = reference["juncs_pred"]
        {
            let size = sizeArray.asArray(Float.self)
            let field = HATField(rawOutput: trace["output"]!)
            let (junctions, lines) = WireframeDecoder.decode(
                field: field,
                imageSize: (width: Int(size[0]), height: Int(size[1])),
                distanceThreshold: network.configuration.distanceThreshold,
                options: .default)

            print("\ndetections")
            print("  junctions  swift \(junctions.count)  torch \(referenceJunctions.dim(0))")
            print("  lines      swift \(lines.count)  torch \(referenceLines.dim(0))")

            let referenceKept = referenceScores.asArray(Float.self).filter { $0 >= 10 }.count
            let keptLines = lines.filter { $0.score >= 10 }
            print("  score>=10  swift \(keptLines.count)  torch \(referenceKept)")

            let expectedJunctions = referenceJunctions.asArray(Float.self)
            let expectedJunctionCount = referenceJunctions.dim(0)
            var junctionsBeyondTolerance = 0
            for junction in junctions {
                var best = Float.greatestFiniteMagnitude
                for index in 0 ..< expectedJunctionCount {
                    let distance = Swift.max(
                        Swift.abs(junction.x - expectedJunctions[index * 2]),
                        Swift.abs(junction.y - expectedJunctions[index * 2 + 1]))
                    if distance < best {
                        best = distance
                        if best == 0 { break }
                    }
                }
                if best > 0.01 { junctionsBeyondTolerance += 1 }
            }
            let junctionAgreement =
                Float(junctions.count - junctionsBeyondTolerance) / Float(Swift.max(junctions.count, 1))
            print(
                "  junctions matched within 0.01 px: "
                    + "\(junctions.count - junctionsBeyondTolerance)/\(junctions.count) "
                    + String(format: "(%.2f%%)", junctionAgreement * 100))

            // Junction order is only defined up to ties in the heatmap score, and the junction
            // *index* is what identifies a line's endpoints — so a tie-flip legitimately
            // permutes both lists. Compare canonically ordered sets instead.
            func canonical(_ rows: [[Float]]) -> [[Float]] {
                rows.sorted { lhs, rhs in
                    for (a, b) in zip(lhs, rhs) where a != b { return a < b }
                    return false
                }
            }

            // A single extra or missing segment shifts every later element of a sorted list, so
            // match each segment to its nearest counterpart instead of comparing positionally.
            // A quantised key is not usable here: field noise of ~1e-4 px flips values that sit
            // near a rounding boundary, which reports spurious mismatches.
            let expectedValues = referenceLines.asArray(Float.self)
            let expectedCount = referenceLines.dim(0)
            var worstNearest: Float = 0
            var beyondTolerance = 0

            for line in lines {
                var best = Float.greatestFiniteMagnitude
                for index in 0 ..< expectedCount {
                    let base = index * 4
                    let distance = Swift.max(
                        Swift.max(
                            Swift.abs(line.x1 - expectedValues[base]),
                            Swift.abs(line.y1 - expectedValues[base + 1])),
                        Swift.max(
                            Swift.abs(line.x2 - expectedValues[base + 2]),
                            Swift.abs(line.y2 - expectedValues[base + 3])))
                    if distance < best {
                        best = distance
                        if best == 0 { break }
                    }
                }
                worstNearest = Swift.max(worstNearest, best)
                if best > 0.01 { beyondTolerance += 1 }
            }
            let lineAgreement =
                Float(lines.count - beyondTolerance) / Float(Swift.max(lines.count, 1))
            print(
                "  lines matched within 0.01 px: \(lines.count - beyondTolerance)/\(lines.count) "
                    + String(format: "(%.2f%%)", lineAgreement * 100)
                    + "  worst \(String(format: "%.3e", worstNearest)) px")

            // Junction selection is capped at 512 by score, and the score gap at that cutoff can
            // fall below fp32 noise (it is ~6e-8 for the v2 checkpoint). Which junction makes the
            // cap is then genuinely undetermined, and one swapped junction re-routes the handful
            // of segments that snapped to it. Gate on agreement rate, not exact set equality.
            if lineAgreement < 0.99 || junctionAgreement < 0.99 {
                failures += 1
                print("  detection agreement below 99%")
            }
        }

        if failures > 0 {
            throw ExitCode(1)
        }
        print("\nall stages within \(tolerance) relative error")
    }
}
