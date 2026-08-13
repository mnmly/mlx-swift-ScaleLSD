// swift-tools-version: 6.0
// Port of ScaleLSD (PyTorch) → mlx-swift. Inference-only line-segment detector:
// DPT-hybrid backbone (timm vit_base_r50_s16_384: ResNetV2 stages + ViT-B/16) →
// DPT reassemble/refinenet neck → 9-channel HAT-field multitask head → wireframe decode.
// PORT FROM: https://github.com/ant-research/scalelsd (../../python/scalelsd)

import PackageDescription

// SwiftPM only knows what to do with a `.docc` catalog when the DocC plugin is in play.
// Outside a documentation build it reports the folder as an unhandled file — a warning that
// would surface in every downstream consumer's build — so gate both on the same flag.
let isDocumentationBuild =
    Context.environment["SPI_GENERATE_DOCS"] == "1" || Context.environment["BUILD_DOC"] == "1"

let package = Package(
    name: "mlx-swift-ScaleLSD",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "MLXScaleLSD", targets: ["MLXScaleLSD"]),
        .executable(name: "scalelsd", targets: ["ScaleLSDCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        // The `--use-lsd` rectifier path needs a line-segment detector matching OpenCV's
        // `createLineSegmentDetector(LSD_REFINE_NONE)`. It is a self-contained image-processing
        // algorithm with no MLX dependency, so it lives in its own package.
        .package(url: "https://github.com/mnmly/swift-lsd", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "MLXScaleLSD",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "LSD", package: "swift-lsd"),
            ],
            exclude: isDocumentationBuild ? [] : ["Documentation.docc"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "ScaleLSDCLI",
            dependencies: [
                "MLXScaleLSD",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "LSD", package: "swift-lsd"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // The SwiftUI demo is an Xcode project under Examples/, not an SPM target — it needs
        // a bundle, entitlements and the app sandbox. See Examples/README.md.
        .testTarget(
            name: "MLXScaleLSDTests",
            dependencies: ["MLXScaleLSD"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation, so normal builds and
// downstream consumers of MLXScaleLSD don't have to resolve an extra dependency.
if isDocumentationBuild {
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
