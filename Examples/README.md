# Examples

Demo frontends for `MLXScaleLSD`.

Each demo is declared as an executable target in the package's `Package.swift` rather than as a
separate `.xcodeproj`, so it builds with the same `xcodebuild -scheme` invocation as the CLI and
cannot silently stop compiling against the library.

| demo | what it shows |
|---|---|
| [`ScaleLSDDemo`](ScaleLSDDemo) | macOS SwiftUI app: drop an image, live threshold controls, in-app weight download |

## ScaleLSDDemo

```bash
xcodebuild -scheme ScaleLSDDemo -destination 'platform=macOS' \
    -configuration Release -derivedDataPath .xcdd build
.xcdd/Build/Products/Release/ScaleLSDDemo
```

On first launch it has no weights. Either press **Download Weights** (fetches from
[mnmly/scalelsd-mlx](https://huggingface.co/mnmly/scalelsd-mlx)) or point it at a folder
produced by `Scripts/convert.py`.

It exists to demonstrate the split that `ScaleLSDSession` enforces:

- the **score** slider re-filters a cached array — the network does not run again;
- the **junction** controls re-run only the decoder against the retained HAT field;
- only opening a new image runs the backbone.

All the work happens in `ScaleLSDSession`; the app owns nothing but presentation, cadence and
the file-access grant. Note that it bakes dropped images to pixels immediately — a lazily
decoded `CGImage` would fault later, on the detection task, once the drag grant has lapsed.
