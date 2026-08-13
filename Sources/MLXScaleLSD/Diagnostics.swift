// Stage-by-stage activation capture, used to bisect the port against PyTorch.
//
// See docs/PARITY.md. The names here match the keys written by Scripts/make_fixtures.py.

import Foundation
import MLX

/// Collects named intermediate activations during a forward pass.
///
/// Passing a recorder is what makes a forward pass traceable; a `nil` recorder costs nothing.
public final class TraceRecorder {

    /// Captured activations, keyed by stage name, in NHWC (spatial) or `(B, N, C)` (tokens).
    public private(set) var stages: [String: MLXArray] = [:]

    /// Stage names in the order they were recorded.
    public private(set) var order: [String] = []

    public init() {}

    func record(_ name: String, _ value: MLXArray) {
        if stages[name] == nil { order.append(name) }
        stages[name] = value
    }

    public subscript(name: String) -> MLXArray? { stages[name] }
}

extension ScaleLSD {

    /// Run the backbone while capturing every stage the parity fixtures cover.
    ///
    /// - Parameter x: NHWC image batch, values in `0...1`.
    /// - Returns: the raw 9-channel HAT field and the recorder holding intermediates.
    public func traced(_ x: MLXArray) -> (output: MLXArray, trace: TraceRecorder) {
        let recorder = TraceRecorder()
        var input = x
        if input.dim(-1) == 1 {
            input = concatenated([input, input, input], axis: -1)
        }
        let levels = backbone.pretrained(input, recorder: recorder)
        let output = backbone.scratch(levels, recorder: recorder)
        recorder.record("output", output)
        return (output, recorder)
    }
}
