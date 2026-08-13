// Turning a CGImage into the network's input tensor.
// PORT FROM: predictor/predict.py (cv2.imread(..., 0) -> cv2.resize -> /255)

import CoreGraphics
import Foundation
import MLX

public enum ImageProcessingError: Error, LocalizedError {
    case cannotCreateContext
    case cannotDecode

    public var errorDescription: String? {
        switch self {
        case .cannotCreateContext: return "Could not create a CoreGraphics drawing context."
        case .cannotDecode: return "Could not decode the image into pixels."
        }
    }
}

public enum ImageProcessing {

    /// Decode `image` to a fully materialised, memory-backed copy.
    ///
    /// `CGImage`s produced by `CGImageSource` are lazily decoded: the file is only read on first
    /// pixel access. A sandboxed caller must bake the pixels *while its file grant is still
    /// valid*, otherwise the deferred decode faults once the grant expires. Frontends should
    /// call this at drop/open time and hand the result to ``ScaleLSDSession``.
    public static func bakedCopy(of image: CGImage) throws -> CGImage {
        guard
            let context = CGContext(
                data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ImageProcessingError.cannotCreateContext }
        context.draw(
            image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let baked = context.makeImage() else { throw ImageProcessingError.cannotDecode }
        return baked
    }

    /// Grayscale luma at the image's native resolution, as an NHWC `(1, H, W, 1)` tensor in
    /// `0...1`.
    ///
    /// Uses the BT.601 weights OpenCV applies for `imread(..., IMREAD_GRAYSCALE)` on the raw
    /// sRGB bytes. CoreGraphics' own DeviceGray conversion is colour-managed and would not
    /// match the reference.
    static func grayscale(_ image: CGImage) throws -> MLXArray {
        let width = image.width
        let height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)

        let result: Bool = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard result else { throw ImageProcessingError.cannotCreateContext }

        var luma = [Float](repeating: 0, count: width * height)
        for index in 0 ..< (width * height) {
            let r = Float(pixels[index * 4])
            let g = Float(pixels[index * 4 + 1])
            let b = Float(pixels[index * 4 + 2])
            luma[index] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        return MLXArray(luma, [1, height, width, 1])
    }

    /// Grayscale, resized to `size × size`, ready for the network.
    ///
    /// The resize uses the same `align_corners=False` sample mapping as `cv2.resize`'s
    /// `INTER_LINEAR`, so this matches the reference preprocessing up to 8-bit quantisation.
    static func networkInput(from image: CGImage, size: Int) throws -> MLXArray {
        let gray = try grayscale(image)
        return bilinearResize(gray, height: size, width: size)
    }
}
