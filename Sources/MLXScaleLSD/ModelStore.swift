// Locating and downloading converted ScaleLSD weights. Shared by the CLI and the demo app.
//
// A "model directory" is any folder holding `config.json` + `model.safetensors`, i.e. the
// output of Scripts/convert.py.
//
// NOTE: the upstream HuggingFace repo (`cherubicxn/scalelsd`) ships PyTorch `.pt` pickles,
// which MLX cannot read. `ModelStore.repository` therefore points at a repo of *converted*
// safetensors; publish one with Scripts/convert.py before the download path can work, or
// point a frontend at a local directory instead.

import Foundation

public enum ModelStore {

    /// Available checkpoints.
    public enum Variant: String, Sendable, CaseIterable, Identifiable {
        /// `scalelsd-vitbase-v1-train-sa1b` — no LayerScale.
        case v1
        /// `scalelsd-vitbase-v2-train-sa1b` — LayerScale in every ViT block.
        case v2

        public var id: String { rawValue }

        /// Subdirectory within the repository and the local cache.
        public var directoryName: String { "scalelsd-vitbase-\(rawValue)" }

        public var displayName: String {
            switch self {
            case .v1: return "ScaleLSD v1 (SA-1B)"
            case .v2: return "ScaleLSD v2 (SA-1B, LayerScale)"
            }
        }
    }

    /// Default HuggingFace repository holding converted MLX weights.
    ///
    /// Pass a different one to ``download(_:repository:progress:)`` to override.
    public static let defaultRepository = "mnmly/mlx-swift-scalelsd"

    public static let requiredFiles = ["config.json", "model.safetensors"]

    /// Whether `directory` holds a usable model.
    public static func isValidModelDirectory(_ directory: URL) -> Bool {
        requiredFiles.allSatisfy {
            FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
        }
    }

    /// Accept either the model directory itself or a HuggingFace cache folder wrapping it.
    public static func resolveModelDirectory(pickedAt url: URL) -> URL? {
        if isValidModelDirectory(url) { return url }
        for child in ["snapshots"] {
            let candidate = url.appending(path: child)
            if let entries = try? FileManager.default.contentsOfDirectory(
                at: candidate, includingPropertiesForKeys: nil),
                let match = entries.first(where: isValidModelDirectory)
            {
                return match
            }
        }
        // A folder of variants, e.g. <picked>/scalelsd-vitbase-v1/.
        for variant in Variant.allCases {
            let candidate = url.appending(path: variant.directoryName)
            if isValidModelDirectory(candidate) { return candidate }
        }
        return nil
    }

    /// Where downloads are cached: `~/Library/Application Support/MLXScaleLSD/<variant>`.
    public static func cacheDirectory(for variant: Variant) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first ?? URL(filePath: NSTemporaryDirectory())
        return base.appending(path: "MLXScaleLSD").appending(path: variant.directoryName)
    }

    /// An already-downloaded copy of `variant`, if present.
    public static func existingModelDirectory(for variant: Variant) -> URL? {
        let cached = cacheDirectory(for: variant)
        return isValidModelDirectory(cached) ? cached : nil
    }

    // MARK: - Download

    public enum DownloadError: Error, LocalizedError {
        case badResponse(URL, Int)

        public var errorDescription: String? {
            switch self {
            case .badResponse(let url, let code):
                return """
                    Download failed with HTTP \(code) for \(url.lastPathComponent).
                    ScaleLSD publishes PyTorch .pt files, which MLX cannot read — this \
                    downloader expects a repository of *converted* safetensors \
                    (\(defaultRepository) by default). Convert a checkpoint with Scripts/convert.py and point \
                    the app at that folder, or pass a different repository.
                    """
            }
        }
    }

    /// Download `variant` into the local cache, reporting 0...1 progress.
    @discardableResult
    public static func download(
        _ variant: Variant,
        repository: String = defaultRepository,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> URL {
        let directory = cacheDirectory(for: variant)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // The weights dominate; give config.json a token share so the bar moves immediately.
        let weights: [String: Double] = ["config.json": 0.01, "model.safetensors": 0.99]
        var completed = 0.0
        for file in requiredFiles {
            let remote = URL(
                string:
                    "https://huggingface.co/\(repository)/resolve/main/\(variant.directoryName)/\(file)"
            )!
            let base = completed
            let share = weights[file] ?? 0
            try await downloadFile(from: remote, to: directory.appending(path: file)) { fraction in
                progress(base + share * fraction)
            }
            completed += share
        }
        progress(1.0)
        return directory
    }

    private static func downloadFile(
        from url: URL, to destination: URL, progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        if let http = response as? HTTPURLResponse, !(200 ..< 300).contains(http.statusCode) {
            throw DownloadError.badResponse(url, http.statusCode)
        }
        let total = response.expectedContentLength

        // Stage in a temporary file so an interrupted download cannot leave a valid-looking
        // model directory behind.
        let staging = destination.appendingPathExtension("partial")
        FileManager.default.createFile(atPath: staging.path, contents: nil)
        let handle = try FileHandle(forWritingTo: staging)

        var buffer = Data()
        buffer.reserveCapacity(1 << 20)
        var received: Int64 = 0
        for try await byte in bytes {
            buffer.append(byte)
            received += 1
            if buffer.count >= (1 << 20) {
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
                if total > 0 { progress(Double(received) / Double(total)) }
            }
        }
        if !buffer.isEmpty { try handle.write(contentsOf: buffer) }
        try handle.close()

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: staging, to: destination)
        progress(1.0)
    }
}
