// Demo view model. Owns only presentation state and cadence — every bit of workload lives in
// `ScaleLSDSession`.

import CoreGraphics
import Foundation
import MLXScaleLSD
import Observation

@MainActor
@Observable
final class DetectorModel {

    enum Phase: Equatable {
        case needsModel
        case downloading(Double)
        case loadingModel
        case ready
        case detecting
        case failed(String)
    }

    private(set) var phase: Phase = .needsModel
    private(set) var modelDirectory: URL?
    private(set) var variant: ModelStore.Variant = .v2

    /// Baked pixels of the current image. Baking happens at open/drop time, while the file
    /// grant is valid — a lazily-decoded CGImage would fault later on the detection task.
    private(set) var image: CGImage?
    private(set) var imageName: String?

    private(set) var result: DetectionResult?
    private(set) var inferenceDuration: Double = 0

    /// Score threshold. Changing it only re-filters a cached array.
    var scoreThreshold: Float = 10 {
        didSet { refilter() }
    }
    /// Junction threshold. Changing it re-runs the decoder, never the network.
    var junctionThreshold: Float = 0.008 {
        didSet { scheduleRedecode() }
    }
    var useNMS: Bool = false {
        didSet { scheduleRedecode() }
    }
    var showJunctions = true

    private(set) var visibleSegments: [LineSegment] = []

    private var session: ScaleLSDSession?
    private var field: FieldHandle?
    private var work: Task<Void, Never>?
    private var redecode: Task<Void, Never>?

    var junctions: [Junction] { showJunctions ? (result?.junctions ?? []) : [] }

    var statusText: String {
        switch phase {
        case .needsModel: return "No model loaded"
        case .downloading(let fraction):
            return "Downloading weights… \(Int(fraction * 100))%"
        case .loadingModel: return "Loading model…"
        case .ready:
            guard let result else { return "Ready — open an image" }
            return "\(visibleSegments.count) of \(result.segments.count) segments · "
                + "\(result.junctions.count) junctions · "
                + String(format: "%.0f ms", inferenceDuration * 1000)
        case .detecting: return "Detecting…"
        case .failed(let message): return message
        }
    }

    // MARK: - Model

    func useExistingModelIfAvailable() {
        // The container cache needs no grant, so prefer it; fall back to a folder the user
        // picked on a previous launch, which does.
        if let existing = ModelStore.existingModelDirectory(for: variant) {
            load(directory: existing)
        } else if let bookmarked = ModelStore.bookmarkedDirectory() {
            load(directory: bookmarked, securityScoped: true)
        }
    }

    func selectVariant(_ newVariant: ModelStore.Variant) {
        guard newVariant != variant else { return }
        variant = newVariant
        session = nil
        field = nil
        result = nil
        visibleSegments = []
        phase = .needsModel
        useExistingModelIfAvailable()
    }

    /// - Parameters:
    ///   - directory: a model directory, or a folder containing one.
    ///   - securityScoped: `true` for a URL the user picked or that came from a bookmark. The
    ///     grant is held across the whole load — the session reads the weights off disk on a
    ///     detached task, so releasing it once the panel closes would fault mid-load.
    func load(directory: URL, securityScoped: Bool = false) {
        let accessing = securityScoped && directory.startAccessingSecurityScopedResource()

        guard let resolved = ModelStore.resolveModelDirectory(pickedAt: directory) else {
            if accessing { directory.stopAccessingSecurityScopedResource() }
            if securityScoped { ModelStore.clearBookmark() }
            phase = .failed("No config.json + model.safetensors in \(directory.lastPathComponent)")
            return
        }
        work?.cancel()
        phase = .loadingModel
        work = Task {
            defer { if accessing { directory.stopAccessingSecurityScopedResource() } }
            do {
                // Loading is blocking and slow; keep it off the main actor.
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try ScaleLSDSession.load(directory: resolved)
                }.value
                guard !Task.isCancelled else { return }
                self.session = loaded
                self.modelDirectory = resolved
                self.phase = .ready
                self.detect()
            } catch {
                guard !Task.isCancelled else { return }
                self.phase = .failed("Could not load model: \(error.localizedDescription)")
            }
        }
    }

    func downloadWeights() {
        work?.cancel()
        phase = .downloading(0)
        work = Task {
            do {
                let directory = try await ModelStore.download(variant) { fraction in
                    Task { @MainActor in
                        if case .downloading = self.phase { self.phase = .downloading(fraction) }
                    }
                }
                guard !Task.isCancelled else { return }
                self.load(directory: directory)
            } catch {
                guard !Task.isCancelled else { return }
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    // MARK: - Image

    /// Adopt an image whose pixels are already materialised.
    func setImage(_ baked: CGImage, name: String?) {
        image = baked
        imageName = name
        field = nil
        result = nil
        visibleSegments = []
        detect()
    }

    // MARK: - Detection

    func detect() {
        guard let session, let image, phase != .detecting else { return }
        work?.cancel()
        phase = .detecting

        let options = currentOptions
        work = Task {
            do {
                // The session is single-writer; this detached task is the only driver.
                let handle = try await Task.detached(priority: .userInitiated) {
                    try session.analyze(image, options: options)
                }.value
                guard !Task.isCancelled else { return }
                let decoded = session.decode(handle, options: options.decoder)

                self.field = handle
                self.result = decoded
                self.inferenceDuration = decoded.inferenceDuration
                self.phase = .ready
                self.refilter()
            } catch {
                guard !Task.isCancelled else { return }
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    private var currentOptions: ScaleLSDOptions {
        ScaleLSDOptions(
            decoder: DecoderOptions(
                junctionThreshold: junctionThreshold, maximumJunctions: 512, useNMS: useNMS))
    }

    /// Cheap: array filter only.
    private func refilter() {
        visibleSegments = result?.segments(minimumScore: scoreThreshold) ?? []
    }

    /// Cheaper than inference: re-runs the decoder against the retained field.
    private func scheduleRedecode() {
        guard let session, let field else { return }
        redecode?.cancel()
        let options = currentOptions.decoder
        redecode = Task {
            // Coalesce slider drags rather than decoding on every intermediate value.
            try? await Task.sleep(for: .milliseconds(60))
            guard !Task.isCancelled else { return }
            // Keep MLX work off the main actor even though a decode is comparatively cheap.
            let decoded = await Task.detached(priority: .userInitiated) {
                session.decode(field, options: options)
            }.value
            guard !Task.isCancelled else { return }
            self.result = decoded
            self.refilter()
        }
    }
}
