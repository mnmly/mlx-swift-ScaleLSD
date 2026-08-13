// `scalelsd` command line entry point.

import ArgumentParser
import Foundation
import MLXScaleLSD

@main
struct ScaleLSDCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scalelsd",
        abstract: "Detect line segments with ScaleLSD on Apple Silicon.",
        subcommands: [Detect.self, Bench.self, Fetch.self, Parity.self]
    )
}
