# ``MLXScaleLSD``

Deep line-segment detection on Apple Silicon.

## Overview

A Swift port of [ScaleLSD](https://github.com/ant-research/scalelsd) built on
[mlx-swift](https://github.com/ml-explore/mlx-swift). Given an image, it returns a *wireframe*:
a set of junctions plus the straight segments connecting them.

The network is a DPT-hybrid — a ResNetV2 trunk feeding a ViT-B/16, then a DPT refinement
decoder — predicting a dense 9-channel HAT (Holistic Attraction) field. Every output pixel
proposes one line segment and votes for a junction; the decoder turns those votes into a
wireframe.

Start with ``ScaleLSDSession``. It is the one entry point that owns weight loading,
preprocessing, inference and decoding, and it is what both the `scalelsd` CLI and the bundled
SwiftUI demo consume.

```swift
let directory = try await ModelStore.download(.v2)
let session = try ScaleLSDSession.load(directory: directory)

let image = try ScaleLSDSession.loadImage(at: imageURL)
let result = try session.detect(image)

for segment in result.segments(minimumScore: 10) {
    print(segment.x1, segment.y1, segment.x2, segment.y2, segment.score)
}
```

### Vanishing points

Detected segments can be grouped by the scene direction they converge on, which is what
distinguishes a facade's horizontals from its verticals:

```swift
let segments = result.segments(minimumScore: 10)
for vanishing in VanishingPointEstimator.estimate(segments: segments) {
    // `nil` when the supporting lines are parallel — a vanishing point at infinity.
    print(vanishing.imagePoint as Any, vanishing.supportingSegments.count)
}
```

This is not part of upstream ScaleLSD — see ``VanishingPointEstimator`` for what it is and is
not verified against.

### Separating the expensive half from the cheap half

The forward pass dominates cost; re-thresholding does not. Two levels of reuse are exposed so a
UI control never re-runs the network:

- Changing the **score** threshold is a pure array filter — ``DetectionResult/segments(minimumScore:)``.
- Changing **junction** thresholds re-runs only the decoder — hold the ``FieldHandle`` from
  ``ScaleLSDSession/analyze(_:options:)`` and call ``ScaleLSDSession/decode(_:options:)`` again.

### Working with pixels, not paths

``ScaleLSDSession`` deliberately takes a decoded `CGImage`. A `CGImage` from `CGImageSource`
decodes lazily, so a sandboxed app must materialise the pixels with
``ImageProcessing/bakedCopy(of:)`` while its file grant is still valid — otherwise the deferred
decode faults later, on the detection task.

## Topics

### Detecting line segments

- ``ScaleLSDSession``
- ``ScaleLSDOptions``
- ``DetectionResult``
- ``LineSegment``
- ``Junction``

### Grouping by scene direction

- ``VanishingPoint``
- ``VanishingPointEstimator``
- ``VanishingPointOptions``

### Tuning the decoder

- ``DecoderOptions``
- ``FieldHandle``
- ``HATField``
- ``WireframeDecoder``

### Obtaining weights

- ``ModelStore``
- ``ScaleLSDConfiguration``
- ``ScaleLSDLoadError``

### Preparing and drawing images

- ``ImageProcessing``
- ``WireframeRenderer``
- ``ImageProcessingError``

### The network itself

Most callers should not need these — they are exposed for custom pipelines and for verifying
the port.

- ``ScaleLSD``
- ``TraceRecorder``
- ``LineFieldEncoder``
