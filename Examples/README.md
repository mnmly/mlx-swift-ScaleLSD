# Examples

Demo frontends for `MLXScaleLSD`.

| demo | what it shows |
|---|---|
| [`ScaleLSDDemo`](ScaleLSDDemo) | macOS SwiftUI app: drop an image, live threshold controls, in-app weight download |

## ScaleLSDDemo

An Xcode project rather than an SPM executable, because it needs what only a real app bundle
gives: an `Info.plist`, entitlements and the App Sandbox. It depends on the package by relative
path (`XCLocalSwiftPackageReference "../.."`), so it always builds against the working tree.

```bash
cd Examples/ScaleLSDDemo
xcodebuild -project ScaleLSDDemo.xcodeproj -scheme ScaleLSDDemo \
    -configuration Release -derivedDataPath .dd build
open .dd/Build/Products/Release/ScaleLSDDemo.app
```

Or just open `ScaleLSDDemo.xcodeproj` and hit run.

The project file is generated from [`project.yml`](ScaleLSDDemo/../project.yml) with
[XcodeGen](https://github.com/yonaskolb/XcodeGen) — edit the YAML and re-run `xcodegen generate`
rather than hand-editing the `.pbxproj`.

### Weights

On first launch the app has no weights. Either press **Download Weights** (fetches from
[mnmly/scalelsd-mlx](https://huggingface.co/mnmly/scalelsd-mlx)) or point it at a folder
produced by `Scripts/convert.py`.

Because the app is sandboxed, its download cache lives in the container
(`~/Library/Containers/com.mnmly.ScaleLSDDemo/Data/Library/Application Support/MLXScaleLSD`),
separate from the CLI's. The two do not share a cache.

### What it demonstrates

The split that `ScaleLSDSession` enforces, made visible:

- the **score** slider re-filters a cached array — the network does not run again;
- the **junction** controls re-run only the decoder against the retained HAT field;
- only opening a new image runs the backbone.

All the work happens in `ScaleLSDSession`; the app owns nothing but presentation, cadence and
the file-access grant. It bakes dropped images to pixels immediately — a lazily decoded
`CGImage` would fault later, on the detection task, once the drag grant has lapsed.

### Entitlements

`com.apple.security.network.client` for the weight download, and
`com.apple.security.files.user-selected.read-only` for opened and dropped images. The demo
never writes to user files.
