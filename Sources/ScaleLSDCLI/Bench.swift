// `scalelsd bench` — throughput and leak check.
//
// Loops the pipeline in-process against one loaded session and watches MLX's active memory.
// Active memory that is flat across iterations means no leak; the much larger "peak" figure is
// MLX's reusable buffer cache, not a leak.

import ArgumentParser
import Foundation
import MLX
import MLXScaleLSD

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark inference and check for memory growth."
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: [.customShort("i"), .long], help: "Image to run repeatedly.")
    var image: String

    @Option(help: "Timed iterations.")
    var iterations: Int = 20

    @Option(help: "Untimed warmup iterations.")
    var warmup: Int = 3

    func run() async throws {
        let session = try ScaleLSDSession.load(directory: URL(filePath: modelOptions.model))
        let options = modelOptions.scaleLSDOptions
        // Decode once: the benchmark measures inference, not image decoding.
        let source = try ScaleLSDSession.loadImage(at: URL(filePath: image))

        func megabytes(_ bytes: Int) -> String {
            String(format: "%.1f MB", Double(bytes) / 1_048_576)
        }

        for _ in 0 ..< warmup {
            _ = try session.detect(source, options: options)
        }

        let activeAfterWarmup = MLX.Memory.activeMemory
        var durations: [Double] = []
        var segmentCount = 0

        for iteration in 0 ..< iterations {
            // autoreleasepool is a frontend concern — the CLI opts in per iteration.
            let elapsed: Double = try autoreleasepool {
                let start = DispatchTime.now().uptimeNanoseconds
                let result = try session.detect(source, options: options)
                segmentCount = result.segments(minimumScore: modelOptions.threshold).count
                return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            }
            durations.append(elapsed)
            if iteration == iterations - 1 || iteration % 5 == 0 {
                print(
                    "  iter \(iteration): \(String(format: "%.3f s", elapsed))"
                        + "  active \(megabytes(MLX.Memory.activeMemory))")
            }
        }

        let activeAtEnd = MLX.Memory.activeMemory
        let sorted = durations.sorted()
        let mean = durations.reduce(0, +) / Double(durations.count)

        print("")
        print("image        \(URL(filePath: image).lastPathComponent) (\(source.width)x\(source.height))")
        print("segments     \(segmentCount) at score >= \(modelOptions.threshold)")
        print("iterations   \(iterations)")
        print("mean         \(String(format: "%.3f s", mean))  (\(String(format: "%.2f", 1 / mean)) it/s)")
        print("median       \(String(format: "%.3f s", sorted[sorted.count / 2]))")
        print("min / max    \(String(format: "%.3f", sorted.first!)) / \(String(format: "%.3f", sorted.last!)) s")
        print("")
        print("active memory after warmup  \(megabytes(activeAfterWarmup))")
        print("active memory at end        \(megabytes(activeAtEnd))")
        let growth = activeAtEnd - activeAfterWarmup
        print("growth over \(iterations) iterations   \(megabytes(growth))")
        print("cache (reusable, not a leak) \(megabytes(MLX.Memory.cacheMemory))")
        print("peak                         \(megabytes(MLX.Memory.peakMemory))")

        // A real leak scales with iteration count; allow a small allocator-noise margin.
        if growth > 8 * 1_048_576 {
            print("\nWARNING: active memory grew by \(megabytes(growth)) — investigate.")
            throw ExitCode(1)
        }
        print("\nactive memory is flat — no leak detected")
    }
}
