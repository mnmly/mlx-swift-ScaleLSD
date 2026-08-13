// SwiftUI demo for mlx-swift-ScaleLSD.
//
// The app owns presentation only: image loading grants, overlay drawing, slider cadence.
// All detection work goes through `ScaleLSDSession`.

import AppKit
import CoreGraphics
import MLXScaleLSD
import SwiftUI
import UniformTypeIdentifiers

@main
struct ScaleLSDDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("ScaleLSD") {
            ContentView()
                .frame(minWidth: 900, minHeight: 600)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

/// Single-window app: quitting with the window is the expected behaviour for a demo.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct ContentView: View {
    @State private var model = DetectorModel()
    @State private var isTargeted = false

    var body: some View {
        HSplitView {
            canvas
                .frame(minWidth: 520)
            InspectorView(model: model)
                .frame(width: 280)
        }
        .background(.background)
        .task { model.useExistingModelIfAvailable() }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    openImage()
                } label: {
                    Label("Open Image", systemImage: "photo")
                }
                .disabled(model.modelDirectory == nil)
            }
        }
    }

    // MARK: - Canvas

    @ViewBuilder
    private var canvas: some View {
        ZStack {
            if let image = model.image {
                if model.isLookingAround, let pose = model.cameraPose {
                    // Always grouped here: the whole reason the 3D view exists is that the
                    // vanishing points resolved, so hiding the grouping would be perverse.
                    LookAroundView(
                        image: image, pose: pose, segments: model.visibleSegments,
                        groups: model.segmentGroups)
                } else {
                    WireframeView(
                        image: image, segments: model.visibleSegments,
                        junctions: model.junctions,
                        groups: model.showVanishingPoints ? model.segmentGroups : [],
                        horizon: model.showHorizon ? model.cameraPose?.horizon : nil)
                }
            } else {
                EmptyStateView(model: model)
            }

            if case .detecting = model.phase {
                ProgressView().controlSize(.large)
                    .padding(24)
                    .background(.regularMaterial, in: .rect(cornerRadius: 12))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .underPageBackgroundColor))
        .overlay {
            if isTargeted {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Color.accentColor, lineWidth: 3)
                    .padding(8)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers)
        }
        .animation(.smooth(duration: 0.2), value: isTargeted)
    }

    // MARK: - Image intake

    /// Bake pixels immediately: a `CGImageSource` image decodes lazily, and the decode would
    /// otherwise happen on the detection task after the file grant has lapsed.
    private func adopt(url: URL) {
        do {
            let baked = try ScaleLSDSession.loadImage(at: url)
            model.setImage(baked, name: url.lastPathComponent)
        } catch {
            NSSound.beep()
        }
    }

    private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            let granted = url.startAccessingSecurityScopedResource()
            defer { if granted { url.stopAccessingSecurityScopedResource() } }
            adopt(url: url)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            Task { @MainActor in adopt(url: url) }
        }
        return true
    }
}

/// The image with its wireframe drawn on top, scaled to fit.
struct WireframeView: View {
    let image: CGImage
    let segments: [LineSegment]
    let junctions: [Junction]
    /// Parallel to `segments`: which vanishing point each supports, or -1. Empty to disable.
    var groups: [Int] = []
    /// Homogeneous line coefficients `(a, b, c)` with `ax + by + c = 0` in pixels, or nil.
    var horizon: SIMD3<Float>?

    /// Matches WireframeRenderer.Style.vanishingPointColors, so the on-screen overlay and an
    /// exported PNG agree.
    private static let groupColors: [Color] = [.red, .green, .blue, .yellow]

    var body: some View {
        GeometryReader { geometry in
            let scale = min(
                geometry.size.width / CGFloat(image.width),
                geometry.size.height / CGFloat(image.height))
            let drawn = CGSize(
                width: CGFloat(image.width) * scale, height: CGFloat(image.height) * scale)
            let origin = CGPoint(
                x: (geometry.size.width - drawn.width) / 2,
                y: (geometry.size.height - drawn.height) / 2)

            Canvas { context, _ in
                context.draw(
                    Image(decorative: image, scale: 1),
                    in: CGRect(origin: origin, size: drawn))

                // One path per colour so each is a single stroke.
                var paths: [Int: Path] = [:]
                for (index, segment) in segments.enumerated() {
                    let group = index < groups.count ? groups[index] : -1
                    paths[group, default: Path()].addLines([
                        CGPoint(
                            x: origin.x + CGFloat(segment.x1) * scale,
                            y: origin.y + CGFloat(segment.y1) * scale),
                        CGPoint(
                            x: origin.x + CGFloat(segment.x2) * scale,
                            y: origin.y + CGFloat(segment.y2) * scale),
                    ])
                }
                for (group, path) in paths {
                    let color =
                        group < 0
                        ? Color.orange
                        : Self.groupColors[group % Self.groupColors.count]
                    context.stroke(
                        path, with: .color(color), style: .init(lineWidth: 1.6, lineCap: .round))
                }

                for junction in junctions {
                    let radius: CGFloat = 2
                    let point = CGPoint(
                        x: origin.x + CGFloat(junction.x) * scale,
                        y: origin.y + CGFloat(junction.y) * scale)
                    context.fill(
                        Path(
                            ellipseIn: CGRect(
                                x: point.x - radius, y: point.y - radius,
                                width: radius * 2, height: radius * 2)),
                        with: .color(.cyan))
                }

                if let horizon,
                    let ends = Self.horizonEnds(
                        horizon, width: CGFloat(image.width), height: CGFloat(image.height))
                {
                    var path = Path()
                    path.addLines([
                        CGPoint(
                            x: origin.x + ends.0.x * scale, y: origin.y + ends.0.y * scale),
                        CGPoint(
                            x: origin.x + ends.1.x * scale, y: origin.y + ends.1.y * scale),
                    ])
                    context.stroke(
                        path, with: .color(.white),
                        style: .init(lineWidth: 1.4, dash: [7, 5]))
                }
            }
        }
    }

    /// Where `ax + by + c = 0` crosses the image rectangle.
    ///
    /// Returns `nil` when the line misses the image — a common outcome for a steeply pitched
    /// camera, where the horizon is genuinely out of frame.
    private static func horizonEnds(
        _ line: SIMD3<Float>, width: CGFloat, height: CGFloat
    ) -> (CGPoint, CGPoint)? {
        let a = CGFloat(line.x)
        let b = CGFloat(line.y)
        let c = CGFloat(line.z)
        var hits: [CGPoint] = []
        if abs(b) > 1e-9 {
            for x in [0, width] {
                let y = -(a * x + c) / b
                if y >= 0, y <= height { hits.append(CGPoint(x: x, y: y)) }
            }
        }
        if abs(a) > 1e-9 {
            for y in [0, height] {
                let x = -(b * y + c) / a
                if x >= 0, x <= width { hits.append(CGPoint(x: x, y: y)) }
            }
        }
        guard hits.count >= 2 else { return nil }
        // A line through a corner is found by both loops, so pick the farthest-apart pair
        // rather than the first two.
        var best = (hits[0], hits[1])
        var bestDistance: CGFloat = -1
        for i in hits.indices {
            for j in (i + 1) ..< hits.count {
                let distance = hypot(hits[i].x - hits[j].x, hits[i].y - hits[j].y)
                if distance > bestDistance {
                    bestDistance = distance
                    best = (hits[i], hits[j])
                }
            }
        }
        return bestDistance > 0 ? best : nil
    }
}

/// Shown before an image is loaded — including the model-acquisition flow.
struct EmptyStateView: View {
    let model: DetectorModel

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "scribble.variable")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(.tertiary)

            switch model.phase {
            case .needsModel:
                Text("Weights required")
                    .font(.title3.weight(.medium))
                Text(
                    "Download the converted checkpoint, or choose a folder produced by "
                        + "Scripts/convert.py."
                )
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 340)

                HStack(spacing: 10) {
                    Button("Download Weights") { model.downloadWeights() }
                        .buttonStyle(.borderedProminent)
                    Button("Choose Folder…") { chooseFolder() }
                }

            case .downloading(let fraction):
                Text("Downloading weights")
                    .font(.title3.weight(.medium))
                ProgressView(value: fraction)
                    .frame(width: 260)

            case .loadingModel:
                Text("Loading model…").font(.title3.weight(.medium))
                ProgressView().controlSize(.small)

            case .failed(let message):
                Text("Something went wrong").font(.title3.weight(.medium))
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
                HStack(spacing: 10) {
                    Button("Retry Download") { model.downloadWeights() }
                    Button("Choose Folder…") { chooseFolder() }
                }

            default:
                Text("Drop an image here")
                    .font(.title3.weight(.medium))
                Text("or use Open Image in the toolbar")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(40)
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            // Persist the grant now, while the panel's access is still live, so the choice
            // survives relaunch. Sandboxed apps otherwise forget it immediately.
            ModelStore.saveBookmark(for: url)
            model.load(directory: url, securityScoped: true)
        }
    }
}

/// Threshold controls. Score is a pure filter; the junction controls re-decode.
struct InspectorView: View {
    @Bindable var model: DetectorModel

    var body: some View {
        Form {
            Section("Model") {
                Picker(
                    "Checkpoint",
                    selection: Binding(
                        get: { model.variant }, set: { model.selectVariant($0) })
                ) {
                    ForEach(ModelStore.Variant.allCases) { variant in
                        Text(variant.displayName).tag(variant)
                    }
                }
                .pickerStyle(.menu)

                if model.modelDirectory == nil {
                    Button("Download Weights") { model.downloadWeights() }
                }
            }

            Section("Segments") {
                LabeledContent("Score ≥ \(Int(model.scoreThreshold))") {
                    Slider(value: $model.scoreThreshold, in: 1 ... 100, step: 1)
                }
                Text("Re-filters instantly — the network does not run again.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Junctions") {
                LabeledContent(String(format: "Heatmap ≥ %.3f", model.junctionThreshold)) {
                    Slider(value: $model.junctionThreshold, in: 0.001 ... 0.2)
                }
                Toggle("Non-maximum suppression", isOn: $model.useNMS)
                Toggle("Show junctions", isOn: $model.showJunctions)
                Text("Re-runs the decoder only, not the backbone.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Vanishing points") {
                Toggle("Group by vanishing point", isOn: $model.showVanishingPoints)
                Text("Colours each segment by the direction it converges on. Derived from the "
                    + "visible segments, so it follows the score threshold.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Camera") {
                if let pose = model.cameraPose, let image = model.image {
                    LabeledContent("Focal length") {
                        Text(String(format: "%.1f px", pose.focalLength))
                    }
                    LabeledContent("Horizontal FOV") {
                        Text(
                            String(
                                format: "%.1f°",
                                pose.horizontalFieldOfView(width: Float(image.width))))
                    }
                    LabeledContent("Roll") { Text(String(format: "%.2f°", pose.roll)) }
                    LabeledContent("Pitch") { Text(String(format: "%.2f°", pose.pitch)) }
                    Toggle("Show horizon", isOn: $model.showHorizon)
                }

                Toggle(
                    "Look around (3D)",
                    isOn: Binding(
                        get: { model.isLookingAround },
                        set: { model.showLookAround = $0 })
                )
                .disabled(model.lookAroundUnavailableReason != nil)

                Text(
                    model.lookAroundUnavailableReason
                        ?? "Rotates the recovered camera in place. A single image has no "
                            + "parallax, so there is nothing to orbit — turning is the only "
                            + "motion the data supports."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

                if model.isLookingAround {
                    Text("Gnomon: cyan and magenta are the two horizontal scene axes, white the "
                        + "vertical one. Drawn as bars because the estimate fixes each axis "
                        + "only up to sign.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Status") {
                Text(model.statusText)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
        .disabled(model.image == nil && model.modelDirectory != nil)
    }
}
