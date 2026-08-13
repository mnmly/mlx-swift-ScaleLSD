// `scalelsd fetch` — download converted weights from the Hub.

import ArgumentParser
import Foundation
import MLXScaleLSD
import Synchronization

struct Fetch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download converted weights into the local cache."
    )

    @Option(help: "Checkpoint to fetch: v1 or v2.")
    var variant: String = "v2"

    @Option(help: "HuggingFace repository holding converted safetensors.")
    var repository: String = ModelStore.defaultRepository

    func run() async throws {
        guard let choice = ModelStore.Variant(rawValue: variant) else {
            throw ValidationError("--variant must be v1 or v2")
        }
        if let existing = ModelStore.existingModelDirectory(for: choice) {
            print("already cached: \(existing.path)")
            return
        }
        // The progress callback is invoked from the download task, so the last-reported
        // decile needs to be shared safely rather than captured as a plain var.
        let lastReported = Mutex(-1)
        let directory = try await ModelStore.download(choice, repository: repository) { fraction in
            let decile = Int(fraction * 10)
            let shouldPrint = lastReported.withLock { last -> Bool in
                guard decile > last else { return false }
                last = decile
                return true
            }
            if shouldPrint { print("  \(decile * 10)%") }
        }
        print("downloaded to \(directory.path)")
    }
}


