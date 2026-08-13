// The photograph re-rendered in 3D, so the recovered camera can be turned in place.
//
// A pure camera rotation induces the homography `H = K R K⁻¹`, which carries no depth term — so
// this is an exact re-render, not an approximation. The construction used here is the equivalent
// one: a textured plane sized to the recovered frustum in front of a perspective camera carrying
// the recovered field of view. Rotating that camera about the optical centre reproduces `H`,
// with the GPU doing the warp.
//
// There is deliberately no way to move the camera. Translation needs parallax, which a single
// image does not contain; an orbit control would be showing structure that was never measured.

import AppKit
import CoreGraphics
import MLXScaleLSD
import RealityKit
import SwiftUI
import simd

/// Drag-to-turn view of a detection, rendered through its recovered camera.
struct LookAroundView: View {
    let image: CGImage
    let pose: CameraPose
    let segments: [LineSegment]
    /// Parallel to `segments`: which vanishing point each supports, or -1.
    let groups: [Int]

    /// Beyond this the view tips towards gimbal flip and the horizon reads as upside-down.
    private static let pitchLimit: Float = 80 * .pi / 180
    private static let fieldOfViewLimits: ClosedRange<Float> = 15 ... 110

    @State private var scene = LookAroundScene()
    @State private var yaw: Float = 0
    @State private var pitch: Float = 0
    @State private var restingYaw: Float = 0
    @State private var restingPitch: Float = 0
    /// `nil` means "the lens we recovered" — the value the reset button returns to.
    @State private var fieldOfView: Float?
    @State private var pinchStart: Float?

    /// The field of view that makes the plane exactly fill the frame.
    private var nativeFieldOfView: Float {
        pose.verticalFieldOfView(height: Float(image.height))
    }

    private var currentFieldOfView: Float { fieldOfView ?? nativeFieldOfView }

    var body: some View {
        // Bind before the closures so `make` — which is `@Sendable` — captures the scene rather
        // than the whole view.
        let scene = scene

        GeometryReader { geometry in
            RealityView { content in
                content.camera = .virtual
                content.add(scene.root)
            } update: { _ in
                scene.sync(image: image, pose: pose, segments: segments, groups: groups)
                scene.look(yaw: yaw, pitch: pitch, fieldOfView: currentFieldOfView)
            }
            // RealityKit's own controls orbit and dolly, which is exactly what this scene
            // must not do.
            .realityViewCameraControls(.none)
            .gesture(turnGesture(viewHeight: geometry.size.height))
            .simultaneousGesture(zoomGesture)
        }
        // The vertical field of view maps onto the view's height, so the view has to carry the
        // image's aspect ratio for the width to come out right too.
        .aspectRatio(CGSize(width: image.width, height: image.height), contentMode: .fit)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // The same colour the scene's backdrop is painted in, so the letterbox around the
        // aspect-fitted viewport and the void beyond the plane read as one surface.
        .background(Color(nsColor: LookAroundScene.backdropColor))
        .overlay(alignment: .bottom) { hint }
        .onChange(of: pose) { resetView() }
        .onChange(of: image) { resetView() }
    }

    // MARK: - Controls

    /// Drag to yaw and pitch. One view height of travel turns the camera by one field of view,
    /// so the image tracks the cursor near the centre of the frame.
    private func turnGesture(viewHeight: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let rate = currentFieldOfView * .pi / 180 / Float(max(viewHeight, 1))
                yaw = restingYaw + Float(value.translation.width) * rate
                pitch = min(
                    max(restingPitch + Float(value.translation.height) * rate,
                        -Self.pitchLimit),
                    Self.pitchLimit)
            }
            .onEnded { _ in
                restingYaw = yaw
                restingPitch = pitch
            }
    }

    /// Pinch to change the field of view. Narrowing it below the recovered value is a crop;
    /// widening it past the recovered value is where the plane's edges come into frame.
    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let start = pinchStart ?? currentFieldOfView
                pinchStart = start
                fieldOfView = min(
                    max(start / Float(value.magnification), Self.fieldOfViewLimits.lowerBound),
                    Self.fieldOfViewLimits.upperBound)
            }
            .onEnded { _ in pinchStart = nil }
    }

    private func resetView() {
        yaw = 0
        pitch = 0
        restingYaw = 0
        restingPitch = 0
        fieldOfView = nil
    }

    private var isAtRest: Bool {
        yaw == 0 && pitch == 0 && fieldOfView == nil
    }

    private var hint: some View {
        HStack(spacing: 10) {
            Text(
                isAtRest
                    ? "Drag to turn the camera · pinch to zoom"
                    : String(
                        format: "yaw %.1f°  pitch %.1f°  vFOV %.1f°",
                        yaw * 180 / .pi, pitch * 180 / .pi, currentFieldOfView))
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)

            if !isAtRest {
                Button("Reset", action: resetView)
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(.regularMaterial, in: .capsule)
        .padding(12)
    }
}

/// Owns the RealityKit entities behind ``LookAroundView``.
///
/// Kept out of the `View` so `update` can diff its inputs: re-uploading the texture on every
/// slider tick would make the score slider unusable.
@MainActor
final class LookAroundScene {

    /// Distance from the camera to the image plane. Arbitrary — the projection depends only on
    /// the ratio of the plane's size to its distance — so one keeps the arithmetic readable.
    private static let planeDistance: Float = 1

    /// Segments sit fractionally nearer the camera than the plane. Scaling a point towards the
    /// optical centre leaves its projection unchanged, so this buys depth priority without
    /// shifting anything on screen by even a fraction of a pixel.
    private static let overlayDepth: Float = planeDistance * 0.997

    /// Stroke width for the segments, in source-image pixels, matching the 2D overlay.
    private static let segmentWidthInPixels: Float = 2

    /// What "no pixels were recorded here" looks like. Shared with the view so the letterbox
    /// around the viewport matches it exactly.
    static let backdropColor = NSColor(srgbRed: 0.09, green: 0.09, blue: 0.10, alpha: 1)

    private static let backdropRadius: Float = 60
    private static let gnomonDistance: Float = 0.5

    /// Same palette as `WireframeView` and `WireframeRenderer.Style.vanishingPointColors`, so a
    /// segment keeps its colour when the canvas switches between 2D and 3D.
    private static let groupColors: [NSColor] = [
        NSColor(srgbRed: 1.0, green: 0.23, blue: 0.19, alpha: 1),
        NSColor(srgbRed: 0.20, green: 0.78, blue: 0.35, alpha: 1),
        NSColor(srgbRed: 0.35, green: 0.56, blue: 1.0, alpha: 1),
        NSColor(srgbRed: 1.0, green: 0.80, blue: 0.0, alpha: 1),
    ]
    private static let ungroupedColor = NSColor(srgbRed: 1.0, green: 0.647, blue: 0.0, alpha: 1)

    /// Gnomon colours, deliberately outside the vanishing-point palette so the scene axes are
    /// not mistaken for another segment group. Ordered as `pose.rotation`'s columns:
    /// horizontal, vertical, horizontal.
    private static let gnomonColors: [NSColor] = [
        NSColor(srgbRed: 0.0, green: 0.85, blue: 0.95, alpha: 1),
        NSColor(srgbRed: 0.95, green: 0.95, blue: 0.95, alpha: 1),
        NSColor(srgbRed: 0.95, green: 0.30, blue: 0.85, alpha: 1),
    ]

    /// Everything in the scene, so the view adds one entity and never reaches inside.
    let root = Entity()

    private let camera = PerspectiveCamera()
    private let plane = ModelEntity()
    private let overlay = Entity()
    private let gnomon = Entity()

    /// Rotation taking the gnomon's local axes onto the recovered scene axes, in world space.
    private var sceneAxes = simd_quatf(angle: 0, axis: [0, 1, 0])
    private var imageAspect: Float = 1

    private var builtImage: CGImage?
    private var builtPose: CameraPose?
    private var builtSegments: [LineSegment] = []
    private var builtGroups: [Int] = []

    init() {
        root.addChild(camera)
        root.addChild(plane)
        root.addChild(overlay)
        // The gnomon rides with the camera so it stays in the corner of the frame; only its
        // orientation is anchored to the world.
        camera.addChild(gnomon)

        // Turning past the original frame shows pixels that were never recorded. A plain shell
        // makes that read as "nothing here" rather than as a rendering failure.
        let backdrop = ModelEntity(
            mesh: .generateSphere(radius: Self.backdropRadius),
            materials: [Self.unlit(Self.backdropColor)])
        root.addChild(backdrop)
    }

    // MARK: - Building

    /// Rebuild whatever the new inputs invalidate, and nothing else.
    func sync(image: CGImage, pose: CameraPose, segments: [LineSegment], groups: [Int]) {
        let geometryChanged = builtImage !== image || builtPose != pose
        if geometryChanged {
            buildPlane(image: image, pose: pose)
            buildGnomon(pose: pose)
            builtImage = image
            builtPose = pose
        }
        // Segment world positions are derived from the pose, so a pose change invalidates them
        // even when the segment array is untouched.
        if geometryChanged || segments != builtSegments || groups != builtGroups {
            buildSegments(segments, groups: groups, pose: pose)
            builtSegments = segments
            builtGroups = groups
        }
    }

    private func buildPlane(image: CGImage, pose: CameraPose) {
        let width = Float(image.width)
        let height = Float(image.height)
        imageAspect = width / height

        // The four image corners back-projected onto the plane. Because the camera's vertical
        // field of view is `pose.verticalFieldOfView(height:)` — the same focal length that
        // produced these rays — this quad's silhouette is exactly the viewport, and the render
        // at identity orientation is the original photograph.
        let corners = [
            Self.point(pose: pose, x: 0, y: 0, depth: Self.planeDistance),
            Self.point(pose: pose, x: width, y: 0, depth: Self.planeDistance),
            Self.point(pose: pose, x: width, y: height, depth: Self.planeDistance),
            Self.point(pose: pose, x: 0, y: height, depth: Self.planeDistance),
        ]

        var descriptor = MeshDescriptor(name: "imagePlane")
        descriptor.positions = MeshBuffers.Positions(corners)
        descriptor.normals = MeshBuffers.Normals(
            [SIMD3<Float>](repeating: [0, 0, 1], count: 4))
        // RealityKit samples with v = 0 at the *bottom* of the texture, so the corners run in
        // the opposite vertical order to the image rows.
        descriptor.textureCoordinates = MeshBuffers.TextureCoordinates([
            [0, 1], [1, 1], [1, 0], [0, 0],
        ])
        descriptor.primitives = .triangles([0, 3, 2, 0, 2, 1])

        // The pixels are baked at open time, so this only fails on a format Metal cannot take —
        // in which case the plane stays empty and the backdrop shows through.
        guard
            let mesh = try? MeshResource.generate(from: [descriptor]),
            let texture = try? TextureResource(image: image, options: .init(semantic: .color))
        else {
            plane.components.remove(ModelComponent.self)
            return
        }

        // Unlit and untone-mapped: the point is to reproduce the photograph, not to relight it.
        var material = UnlitMaterial(applyPostProcessToneMap: false)
        material.color = .init(tint: .white, texture: .init(texture))
        material.faceCulling = .none
        plane.model = ModelComponent(mesh: mesh, materials: [material])
    }

    private func buildSegments(_ segments: [LineSegment], groups: [Int], pose: CameraPose) {
        overlay.children.removeAll()

        // A world-space width that corresponds to a fixed number of source pixels, so a segment
        // covers the same pixels it does in the 2D overlay.
        let halfWidth = Self.segmentWidthInPixels * Self.overlayDepth / (2 * pose.focalLength)

        // One mesh per colour: a few hundred segments as individual entities would be a few
        // hundred draw calls.
        var quads: [Int: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
        for (index, segment) in segments.enumerated() {
            let start = Self.point(
                pose: pose, x: segment.x1, y: segment.y1, depth: Self.overlayDepth)
            let end = Self.point(
                pose: pose, x: segment.x2, y: segment.y2, depth: Self.overlayDepth)

            // The overlay lies at constant z, so widening it is a 2D perpendicular.
            var offset = SIMD3<Float>(-(end.y - start.y), end.x - start.x, 0)
            let norm = simd_length(offset)
            guard norm > 1e-9 else { continue }
            offset *= halfWidth / norm

            let group = index < groups.count ? groups[index] : -1
            var quad = quads[group] ?? ([], [])
            let base = UInt32(quad.positions.count)
            quad.positions.append(
                contentsOf: [start - offset, start + offset, end + offset, end - offset])
            quad.indices.append(
                contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
            quads[group] = quad
        }

        for (group, quad) in quads {
            var descriptor = MeshDescriptor(name: "segments\(group)")
            descriptor.positions = MeshBuffers.Positions(quad.positions)
            descriptor.normals = MeshBuffers.Normals(
                [SIMD3<Float>](repeating: [0, 0, 1], count: quad.positions.count))
            descriptor.primitives = .triangles(quad.indices)
            guard let mesh = try? MeshResource.generate(from: [descriptor]) else { continue }

            let color =
                group < 0
                ? Self.ungroupedColor
                : Self.groupColors[group % Self.groupColors.count]
            overlay.addChild(ModelEntity(mesh: mesh, materials: [Self.unlit(color)]))
        }
    }

    private func buildGnomon(pose: CameraPose) {
        gnomon.children.removeAll()

        // `pose.rotation`'s columns are the scene axes in camera coordinates, which are y-down
        // and z-forward; RealityKit's are y-up and z-back. Flipping y and z on each column is a
        // 180° turn about x, so the frame stays right-handed and orthonormal.
        let columns = pose.rotation.columns
        sceneAxes = simd_quatf(
            simd_float3x3(
                SIMD3(columns.0.x, -columns.0.y, -columns.0.z),
                SIMD3(columns.1.x, -columns.1.y, -columns.1.z),
                SIMD3(columns.2.x, -columns.2.y, -columns.2.z)))

        // Bars rather than arrows: the sign of each recovered axis is gauge freedom, so pointing
        // one way would assert more than the estimate supports. Unit-length here; `look` scales
        // the parent to suit the field of view.
        for axis in 0 ..< 3 {
            var size = SIMD3<Float>(repeating: 0.06)
            size[axis] = 1
            gnomon.addChild(
                ModelEntity(
                    mesh: .generateBox(size: size),
                    materials: [Self.unlit(Self.gnomonColors[axis])]))
        }
    }

    // MARK: - Camera

    /// Point the camera and size the gnomon to match.
    func look(yaw: Float, pitch: Float, fieldOfView: Float) {
        // Yaw about world up, then pitch about the camera's own right: composing in that order
        // is what keeps the horizon level, i.e. introduces no roll.
        camera.orientation =
            simd_quatf(angle: yaw, axis: [0, 1, 0]) * simd_quatf(angle: pitch, axis: [1, 0, 0])

        var component = camera.camera
        component.fieldOfViewOrientation = .vertical
        component.fieldOfViewInDegrees = fieldOfView
        camera.camera = component

        // Park the gnomon in the bottom-left of the frame. Where that is depends on the field of
        // view, so it has to follow zoom.
        let halfHeight = Self.gnomonDistance * tan(fieldOfView * .pi / 360)
        gnomon.position = SIMD3(
            -0.78 * halfHeight * imageAspect, -0.72 * halfHeight, -Self.gnomonDistance)
        gnomon.scale = SIMD3(repeating: 0.24 * halfHeight)
        // World-relative: the scene axes are fixed in the world, so the gnomon turns against the
        // camera it is parented to.
        gnomon.setOrientation(sceneAxes, relativeTo: nil)
    }

    // MARK: - Geometry

    /// A source pixel back-projected onto a plane `depth` in front of the camera, in RealityKit
    /// world coordinates.
    ///
    /// ``CameraPose/ray(through:)`` returns a unit ray in camera coordinates — x right, y *down*,
    /// z forward — so scaling it to `z = depth` lands on the plane, and negating y and z converts
    /// to RealityKit's frame, where y is up and the camera looks down -z.
    private static func point(
        pose: CameraPose, x: Float, y: Float, depth: Float
    ) -> SIMD3<Float> {
        let ray = pose.ray(through: SIMD2(x, y))
        let scaled = ray * (depth / ray.z)
        return SIMD3(scaled.x, -scaled.y, -scaled.z)
    }

    private static func unlit(_ color: NSColor) -> UnlitMaterial {
        var material = UnlitMaterial(applyPostProcessToneMap: false)
        material.color = .init(tint: color)
        // Nothing here is a closed solid, and the backdrop is viewed from inside.
        material.faceCulling = .none
        return material
    }
}
