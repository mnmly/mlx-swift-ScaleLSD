// TF-style "SAME" padding, as used by timm's `Conv2dSame` / `MaxPool2dSame`.
// PORT FROM: timm/layers/padding.py

import Foundation
import MLX
import MLXNN

/// TF-compatible SAME padding for one axis.
///
/// Returns the `(before, after)` split for an input of `size` under `kernelSize`/`stride`.
/// The split is deliberately *asymmetric* when the total padding is odd — the extra pixel
/// goes on the trailing edge, matching `timm`'s `pad_same`. Getting this backwards shifts
/// every downstream feature by a pixel.
@inlinable
func samePadding(size: Int, kernelSize: Int, stride: Int, dilation: Int = 1) -> (Int, Int) {
    let outSize = (size + stride - 1) / stride  // ceil(size / stride)
    let total = max((outSize - 1) * stride + (kernelSize - 1) * dilation + 1 - size, 0)
    return (total / 2, total - total / 2)
}

/// Whether SAME padding for these parameters can be expressed as a static symmetric pad.
///
/// Mirrors timm's `is_static_pad`: only stride-1 convolutions with an odd effective kernel
/// qualify. Everything else needs the input-size-dependent asymmetric pad above.
@inlinable
func isStaticSamePadding(kernelSize: Int, stride: Int, dilation: Int = 1) -> Bool {
    stride == 1 && (dilation * (kernelSize - 1)) % 2 == 0
}

/// Apply asymmetric SAME padding to an NHWC tensor.
@inlinable
func padSame(
    _ x: MLXArray, kernelSize: Int, stride: Int, dilation: Int = 1, value: Float = 0
) -> MLXArray {
    let (top, bottom) = samePadding(
        size: x.dim(1), kernelSize: kernelSize, stride: stride, dilation: dilation)
    let (left, right) = samePadding(
        size: x.dim(2), kernelSize: kernelSize, stride: stride, dilation: dilation)
    if top | bottom | left | right == 0 { return x }
    let widths: [IntOrPair] = [
        .init((0, 0)), .init((top, bottom)), .init((left, right)), .init((0, 0)),
    ]
    return padded(x, widths: widths, mode: .constant, value: MLXArray(value).asType(x.dtype))
}

/// A convolution with TF-style SAME padding and pre-standardised weights.
///
/// Upstream uses timm's `StdConv2dSame`, which re-standardises its weight on *every*
/// forward pass. Inference weights are frozen, so `Scripts/convert.py` bakes the
/// standardisation into the stored tensor and this layer is a plain convolution — it only
/// has to reproduce the padding behaviour.
///
/// Weight layout is MLX-native `(O, kH, kW, I)`.
final class SameConv2d: Module, UnaryLayer {

    let weight: MLXArray
    let bias: MLXArray?

    let kernelSize: Int
    let stride: Int
    /// Symmetric padding applied directly by the convolution (static case).
    let staticPadding: Int
    /// Whether the input-size-dependent asymmetric pad must be applied first.
    let dynamicSame: Bool

    init(
        inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int = 1,
        bias: Bool = false
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.dynamicSame = !isStaticSamePadding(kernelSize: kernelSize, stride: stride)
        self.staticPadding = dynamicSame ? 0 : (kernelSize - 1) / 2
        self.weight = MLXArray.zeros([outputChannels, kernelSize, kernelSize, inputChannels])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if dynamicSame {
            h = padSame(h, kernelSize: kernelSize, stride: stride)
        }
        h = conv2d(h, weight, stride: .init(stride), padding: .init(staticPadding))
        if let bias { h = h + bias }
        return h
    }
}

/// Max pooling with TF-style SAME padding (`timm`'s `MaxPool2dSame`).
///
/// `MLXNN.MaxPool2d` only takes symmetric padding, so the asymmetric pad is applied
/// explicitly with `-infinity` — the identity for a max reduction.
final class MaxPool2dSame: Module, UnaryLayer {

    let kernelSize: Int
    let stride: Int
    private let pool: MaxPool2d

    init(kernelSize: Int, stride: Int) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.pool = MaxPool2d(kernelSize: .init(kernelSize), stride: .init(stride))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        pool(padSame(x, kernelSize: kernelSize, stride: stride, value: -.infinity))
    }
}
