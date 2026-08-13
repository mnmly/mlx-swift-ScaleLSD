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

/// Junction non-maximum suppression. Guarded by tests because the default path disables it,
/// so a break here is otherwise only visible to someone toggling it in the GUI.
struct SuppressionTests {

    /// Build a HATField whose junction heatmap is `values`, laid out (H, W).
    private func field(_ values: [[Float]]) -> HATField {
        let h = values.count
        let w = values[0].count
        // Channel 5/6 are the junction logits; softmax over them must yield `values` in
        // channel 6, so set logit 5 to 0 and logit 6 to logit(v).
        var raw = [Float](repeating: 0, count: h * w * 9)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let v = min(max(values[y][x], 1e-6), 1 - 1e-6)
                raw[(y * w + x) * 9 + 6] = log(v / (1 - v))
            }
        }
        return HATField(rawOutput: MLXArray(raw, [1, h, w, 9]))
    }

    @Test("Suppression preserves the input shape")
    func preservesShape() {
        let input = field(Array(repeating: Array(repeating: Float(0.1), count: 16), count: 16))
        let suppressed = input.suppressedJunctionHeatmap(kernelSize: 3)
        eval(suppressed)
        #expect(suppressed.shape == input.junctionHeatmap.shape)
    }

    @Test("Suppression keeps local maxima and zeroes their neighbours")
    func keepsPeaks() {
        var values = Array(repeating: Array(repeating: Float(0.01), count: 9), count: 9)
        values[4][4] = 0.9  // isolated peak
        values[4][5] = 0.5  // neighbour, must be suppressed
        values[1][1] = 0.7  // a second, separated peak

        let suppressed = field(values).suppressedJunctionHeatmap(kernelSize: 3)
        eval(suppressed)
        let out = suppressed.asArray(Float.self)

        #expect(out[4 * 9 + 4] > 0.5, "the peak should survive")
        #expect(out[1 * 9 + 1] > 0.5, "a separated peak should survive")
        #expect(out[4 * 9 + 5] == 0, "a neighbour of a stronger peak should be suppressed")
    }

    @Test("kernelSize 1 is the identity")
    func identity() {
        let input = field([[0.2, 0.4], [0.6, 0.1]])
        let suppressed = input.suppressedJunctionHeatmap(kernelSize: 1)
        eval(suppressed)
        #expect(
            suppressed.asArray(Float.self) == input.junctionHeatmap.asArray(Float.self))
    }
}
