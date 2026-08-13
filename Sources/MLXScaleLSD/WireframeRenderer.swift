// Drawing a detected wireframe over the source image.
// PORT FROM: scalelsd/base/show/painters.py (HAWPainter.draw_wireframe)
//
// Pure CoreGraphics — no SwiftUI — so both the CLI's PNG export and a GUI's "save annotated
// image" can share it. Live on-screen drawing belongs to the frontend.

import CoreGraphics
import Foundation

#if canImport(UniformTypeIdentifiers)
    import ImageIO
    import UniformTypeIdentifiers
#endif

/// Renders wireframes over images.
public enum WireframeRenderer {

    /// Colours matching the upstream painter's defaults.
    public struct Style: Sendable {
        public var edgeColor: CGColor
        public var vertexColor: CGColor
        public var lineWidth: CGFloat
        public var vertexRadius: CGFloat
        /// White wash over the source image, as upstream's `--whitebg` does. 0 disables it.
        public var whiteOverlay: CGFloat

        public init(
            edgeColor: CGColor = CGColor(red: 1.0, green: 0.647, blue: 0.0, alpha: 1.0),
            vertexColor: CGColor = CGColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 1.0),
            lineWidth: CGFloat = 2.0,
            vertexRadius: CGFloat = 2.0,
            whiteOverlay: CGFloat = 0
        ) {
            self.edgeColor = edgeColor
            self.vertexColor = vertexColor
            self.lineWidth = lineWidth
            self.vertexRadius = vertexRadius
            self.whiteOverlay = whiteOverlay
        }

        public static let `default` = Style()
    }

    /// Draw `segments` and `junctions` over `image`.
    ///
    /// Detection coordinates are top-left origin; CoreGraphics is bottom-left, so the context is
    /// flipped before drawing.
    public static func render(
        image: CGImage,
        segments: [LineSegment],
        junctions: [Junction],
        style: Style = .default
    ) throws -> CGImage {
        let width = image.width
        let height = image.height
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ImageProcessingError.cannotCreateContext }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        if style.whiteOverlay > 0 {
            context.setFillColor(
                CGColor(red: 1, green: 1, blue: 1, alpha: style.whiteOverlay))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        }

        // Detection space is top-left origin.
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        context.setStrokeColor(style.edgeColor)
        context.setLineWidth(style.lineWidth)
        context.setLineCap(.round)
        for segment in segments {
            context.move(to: CGPoint(x: CGFloat(segment.x1), y: CGFloat(segment.y1)))
            context.addLine(to: CGPoint(x: CGFloat(segment.x2), y: CGFloat(segment.y2)))
        }
        context.strokePath()

        context.setFillColor(style.vertexColor)
        for junction in junctions {
            let radius = style.vertexRadius
            context.fillEllipse(
                in: CGRect(
                    x: CGFloat(junction.x) - radius, y: CGFloat(junction.y) - radius,
                    width: radius * 2, height: radius * 2))
        }

        guard let output = context.makeImage() else { throw ImageProcessingError.cannotDecode }
        return output
    }

    #if canImport(UniformTypeIdentifiers)
        /// Write a `CGImage` to disk as PNG.
        public static func writePNG(_ image: CGImage, to url: URL) throws {
            guard
                let destination = CGImageDestinationCreateWithURL(
                    url as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { throw ImageProcessingError.cannotCreateContext }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw ImageProcessingError.cannotDecode
            }
        }
    #endif
}
