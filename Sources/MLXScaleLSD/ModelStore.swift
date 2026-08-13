// Locating and downloading converted ScaleLSD weights. Shared by the CLI and the demo app.
//
// A "model directory" is any folder holding `config.json` + `model.safetensors`, i.e. the
// output of Scripts/convert.py.
//
// NOTE: the upstream HuggingFace repo (`cherubicxn/scalelsd`) ships PyTorch `.pt` pickles,
// which MLX cannot read. `defaultRepository` therefore points at a repo of *converted*
// safetensors produced by Scripts/convert.py.

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
    public static let defaultRepository = "mnmly/scalelsd-mlx"

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

    // MARK: - Security-scoped bookmarks

    private static let bookmarkKey = "MLXScaleLSD.modelDirectoryBookmark"

    /// Persist access to a user-chosen model directory across launches.
    ///
    /// A sandboxed app only holds access to a folder the user picked for as long as that grant
    /// lives; without a bookmark the choice is forgotten on relaunch. Call this while access is
    /// still valid.
    public static func saveBookmark(for url: URL) {
        #if os(macOS)
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            let data = try? url.bookmarkData(
                options: [.withSecurityScope], includingResourceValuesForKeys: nil,
                relativeTo: nil)
        #else
            let data = try? url.bookmarkData()
        #endif
        guard let data else { return }
        UserDefaults.standard.set(data, forKey: bookmarkKey)
    }

    /// The previously bookmarked model directory, if one was saved and still resolves.
    ///
    /// The returned URL is security-scoped: call `startAccessingSecurityScopedResource()` before
    /// touching it and stop when done. Returns `nil` if the bookmark is stale or absent.
    public static func bookmarkedDirectory() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var stale = false
        #if os(macOS)
            let url = try? URL(
                resolvingBookmarkData: data, options: [.withSecurityScope], relativeTo: nil,
                bookmarkDataIsStale: &stale)
        #else
            let url = try? URL(
                resolvingBookmarkData: data, relativeTo: nil, bookmarkDataIsStale: &stale)
        #endif
        return stale ? nil : url
    }

    /// Forget a saved bookmark, e.g. after it fails to resolve to a model.
    public static func clearBookmark() {
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
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
        // A `URLSession.bytes` loop awaits per byte, which caps throughput at a few MB/s —
        // roughly six minutes for a 490 MB checkpoint. A download task streams to disk at
        // link speed and reports progress through its delegate.
        let coordinator = DownloadCoordinator(progress: progress)
        let session = URLSession(
            configuration: .default, delegate: coordinator, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let downloaded = try await coordinator.download(url, in: session)

        // Stage under the final name so an interrupted download cannot leave a
        // valid-looking model directory behind.
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: downloaded, to: destination)
        progress(1.0)
    }
}

/// Bridges `URLSessionDownloadTask` progress and completion into async/await.
private final class DownloadCoordinator: NSObject, URLSessionDownloadDelegate, @unchecked Sendable
{
    private let progress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<URL, Error>?

    init(progress: @escaping @Sendable (Double) -> Void) {
        self.progress = progress
    }

    func download(_ url: URL, in session: URLSession) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock { self.continuation = continuation }
            session.downloadTask(with: url).resume()
        }
    }

    private func finish(_ result: Result<URL, Error>) {
        let pending = lock.withLock { () -> CheckedContinuation<URL, Error>? in
            defer { continuation = nil }
            return continuation
        }
        pending?.resume(with: result)
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        progress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        if let response = downloadTask.response as? HTTPURLResponse,
            !(200 ..< 300).contains(response.statusCode)
        {
            finish(.failure(ModelStore.DownloadError.badResponse(
                downloadTask.originalRequest?.url ?? location, response.statusCode)))
            return
        }
        // `location` is deleted as soon as this callback returns, so claim it now.
        let claimed = FileManager.default.temporaryDirectory
            .appending(path: "scalelsd-\(UUID().uuidString)")
        do {
            try FileManager.default.moveItem(at: location, to: claimed)
            finish(.success(claimed))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
    ) {
        if let error { finish(.failure(error)) }
    }
}
