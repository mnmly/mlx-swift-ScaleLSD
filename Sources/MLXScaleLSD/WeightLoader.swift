// Loading a converted ScaleLSD checkpoint.
//
// Scripts/convert.py already performs every layout change (weight standardisation baked in,
// conv+BatchNorm folded, `nn.Sequential` indices named, conv weights transposed to NHWC), so
// loading is a straight `update` with strict verification.

import Foundation
import MLX
import MLXNN

/// Errors raised while locating or loading a converted checkpoint.
public enum ScaleLSDLoadError: Error, LocalizedError {
    case missingFile(URL)
    case parameterCountMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let url):
            return """
                Missing \(url.lastPathComponent) at \(url.path).
                Convert an upstream checkpoint first:
                  python Scripts/convert.py -c scalelsd-vitbase-v2-train-sa1b.pt -o <dir>
                """
        case .parameterCountMismatch(let expected, let actual):
            return "Loaded \(actual) parameters but the model declares \(expected)."
        }
    }
}

extension ScaleLSD {

    /// Total number of scalar parameters currently held by the module tree.
    public var parameterCount: Int {
        parameters().flattenedValues().reduce(0) { $0 + $1.size }
    }

    /// Build a model from a directory containing `config.json` and `model.safetensors`.
    ///
    /// - Parameters:
    ///   - directory: output directory of `Scripts/convert.py`.
    ///   - dtype: weight precision. `.float32` is the default because the HAT-field decode
    ///     is sensitive to precision in the junction-offset channels.
    public static func load(directory: URL, dtype: DType = .float32) throws -> ScaleLSD {
        let configURL = directory.appending(path: "config.json")
        let weightsURL = directory.appending(path: "model.safetensors")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ScaleLSDLoadError.missingFile(configURL)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw ScaleLSDLoadError.missingFile(weightsURL)
        }

        let configuration = try ScaleLSDConfiguration.load(contentsOf: configURL)
        let model = ScaleLSD(configuration: configuration)
        try model.loadWeights(contentsOf: weightsURL, dtype: dtype)
        return model
    }

    /// Load converted weights into an already-built model.
    ///
    /// Verification is `.all` — unused checkpoint keys, unset model keys and shape mismatches
    /// are all fatal. That strictness is what catches converter key-mapping mistakes.
    public func loadWeights(contentsOf url: URL, dtype: DType = .float32) throws {
        let raw = try loadArrays(url: url)
        var weights: [(String, MLXArray)] = []
        weights.reserveCapacity(raw.count)
        for (key, value) in raw {
            weights.append((key, value.asType(dtype)))
        }
        try update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(self)
    }
}
