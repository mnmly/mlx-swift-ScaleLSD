import Foundation
import MLX
import Testing

@testable import MLXScaleLSD

/// Shape-level checks that need no checkpoint.
struct ShapeTests {

    @Test("SAME padding splits the odd remainder onto the trailing edge")
    func samePaddingIsAsymmetric() {
        // stem: 512 input, 7×7 stride 2 -> total pad 5
        #expect(samePadding(size: 512, kernelSize: 7, stride: 2) == (2, 3))
        // stage conv2 / max pool: even input, 3×3 stride 2 -> total pad 1
        #expect(samePadding(size: 128, kernelSize: 3, stride: 2) == (0, 1))
        #expect(samePadding(size: 256, kernelSize: 3, stride: 2) == (0, 1))
        // stride 1 3×3 is symmetric
        #expect(samePadding(size: 128, kernelSize: 3, stride: 1) == (1, 1))
    }

    @Test("Only stride-1 odd kernels use static padding")
    func staticPaddingClassification() {
        #expect(isStaticSamePadding(kernelSize: 3, stride: 1))
        #expect(isStaticSamePadding(kernelSize: 1, stride: 1))
        #expect(!isStaticSamePadding(kernelSize: 7, stride: 2))
        #expect(!isStaticSamePadding(kernelSize: 3, stride: 2))
        #expect(!isStaticSamePadding(kernelSize: 1, stride: 2))
    }

    @Test("Bilinear resize hits the exact requested size")
    func exactResize() {
        // 24 -> 32 is the position-embedding case that Upsample's Int() truncation misses.
        let grid = MLXArray.zeros([1, 24, 24, 8])
        let resized = bilinearResize(grid, height: 32, width: 32)
        #expect(resized.shape == [1, 32, 32, 8])
    }

    @Test("Backbone emits a 9-channel field at stride 2")
    func backboneOutputShape() {
        let model = ScaleLSD(configuration: .v1)
        let input = MLXArray.zeros([1, 128, 128, 1])
        let output = model(input)
        eval(output)
        #expect(output.shape == [1, 64, 64, 9])
    }
}
