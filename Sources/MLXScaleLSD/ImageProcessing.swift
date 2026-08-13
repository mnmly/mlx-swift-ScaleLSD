// Turning a CGImage into the network's input tensor.
// PORT FROM: predictor/predict.py (cv2.imread(..., 0) -> cv2.resize -> /255)

import Accelerate
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
    public static func grayscale(_ image: CGImage) throws -> MLXArray {
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

        // A per-pixel Swift loop over a multi-megapixel image is measurable next to the network
        // itself, so the reduction runs through vDSP over the interleaved channels.
        // BT.601 weights on the raw sRGB bytes, matching OpenCV's IMREAD_GRAYSCALE.
        let count = width * height
        var luma = [Float](repeating: 0, count: count)
        var interleaved = [Float](repeating: 0, count: count * 4)
        vDSP.convertElements(of: pixels, to: &interleaved)

        var red: Float = 0.299
        var green: Float = 0.587
        var blue: Float = 0.114

        interleaved.withUnsafeBufferPointer { source in
            luma.withUnsafeMutableBufferPointer { destination in
                guard let base = source.baseAddress, let out = destination.baseAddress else {
                    return
                }
                let n = vDSP_Length(count)
                // Each channel is a stride-4 view of the interleaved buffer. The weights are
                // applied unscaled and the /255 is left to the end: folding it into the
                // constants rounds them differently and perturbs borderline detections.
                vDSP_vsmul(base, 4, &red, out, 1, n)
                vDSP_vsma(base + 1, 4, &green, out, 1, out, 1, n)
                vDSP_vsma(base + 2, 4, &blue, out, 1, out, 1, n)
            }
        }
        // The reference pipeline is 8-bit throughout (`cv2.imread(..., IMREAD_GRAYSCALE)` then
        // `cv2.resize`), so its input is quantised twice. Reproducing that was measured and
        // rejected: rounding luma to 8 bits here moves preprocessing agreement only from
        // 1.406e-02 to 1.385e-02, because the residual is dominated by cv2's *fixed-point
        // resize*, not the luma step — and it made detection counts marginally worse. Staying
        // in float is both more accurate and closer to the reference. See docs/PARITY.md.
        var scale: Float = 255
        vDSP_vsdiv(luma, 1, &scale, &luma, 1, vDSP_Length(count))
        return MLXArray(luma, [1, height, width, 1])
    }

    /// Grayscale, resized to `size × size`, ready for the network.
    ///
    /// The resize uses the same `align_corners=False` sample mapping as `cv2.resize`'s
    /// `INTER_LINEAR`, so this matches the reference preprocessing up to 8-bit quantisation.
    public static func networkInput(from image: CGImage, size: Int) throws -> MLXArray {
        let gray = try grayscale(image)
        return bilinearResize(gray, height: size, width: size)
    }
}
