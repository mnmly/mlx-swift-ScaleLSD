#!/usr/bin/env python3
"""Convert an upstream ScaleLSD PyTorch checkpoint to MLX-layout safetensors.

    uv run --with torch --with safetensors Scripts/convert.py \
        --checkpoint scalelsd-vitbase-v1-train-sa1b.pt --output out/v1

The Swift side deliberately never sees an unfused module, so this script does all
the graph simplification the upstream model only needs because it can also train:

  * **Weight standardization is baked in.** The ResNetV2 hybrid stem/stages use timm's
    `StdConv2dSame`, which standardises its own weight on every forward. Weights are
    frozen at inference, so the standardised weight is stored directly and Swift runs
    a plain `Conv2d`.
  * **conv+BatchNorm pairs are folded.** `ResidualConvUnit_custom` runs `conv(bias=False)
    -> BatchNorm2d`; the pair collapses into one conv with bias.
  * **`nn.Sequential` index keys are flattened** into named submodules, so the Swift
    module tree reads like the architecture rather than like a list.
  * **Conv weights are transposed** from PyTorch `(O, I, kH, kW)` to MLX `(O, kH, kW, I)`.
  * **The 1000-class ImageNet head is dropped** — ScaleLSD never calls it.

Both simplifications are verified to reproduce the upstream output to ~2e-6 relative.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Architecture constants (see docs/PARITY.md)
# ---------------------------------------------------------------------------

# NOT the StdConv2dSame default of 1e-6: timm's hybrid ViT builds its ResNetV2 with
# `conv_layer=partial(StdConv2dSame, eps=1e-8)`. Using 1e-6 shifts the standardised weights
# enough to cost ~2e-3 relative error at the very first stage.
STD_CONV_EPS = 1e-8
BN_EPS = 1e-5  # torch BatchNorm2d default
HEAD_SIZE = [3, 1, 1, 2, 2]  # MultitaskHead output channels, 9 total


def standardize(weight: torch.Tensor, eps: float = STD_CONV_EPS) -> torch.Tensor:
    """Reproduce timm's `F.batch_norm(w.reshape(1, O, -1), training=True)`.

    Per output channel: (w - mean) / sqrt(biased_var + eps).
    """
    out_channels = weight.shape[0]
    flat = weight.reshape(out_channels, -1)
    mean = flat.mean(dim=1, keepdim=True)
    var = flat.var(dim=1, unbiased=False, keepdim=True)
    return ((flat - mean) / torch.sqrt(var + eps)).reshape_as(weight)


def to_nhwc(weight: torch.Tensor) -> torch.Tensor:
    """PyTorch Conv2d (O, I, kH, kW) -> MLX (O, kH, kW, I)."""
    assert weight.dim() == 4, weight.shape
    return weight.permute(0, 2, 3, 1).contiguous()


# Every conv inside the hybrid ResNetV2 backbone is a StdConv2dSame.
_IS_STD_CONV = re.compile(
    r"^backbone\.pretrained\.model\.patch_embed\.backbone\..*conv\.weight$"
    r"|^backbone\.pretrained\.model\.patch_embed\.backbone\.stages\..*\.conv\d\.weight$"
)


def is_std_conv(key: str) -> bool:
    prefix = "backbone.pretrained.model.patch_embed.backbone."
    if not key.startswith(prefix) or not key.endswith(".weight"):
        return False
    tail = key[len(prefix):]
    return (
        tail == "stem.conv.weight"
        or tail.endswith("downsample.conv.weight")
        or re.search(r"\.conv[123]\.weight$", tail) is not None
    )


def remap_key(key: str) -> str | None:
    """Map an upstream state-dict key to this port's module tree.

    Returns None for keys that are intentionally dropped.
    """
    # The ImageNet classifier is never called by ScaleLSD.
    if key.startswith("backbone.pretrained.model.head."):
        return None
    if key.endswith("num_batches_tracked"):
        return None

    k = key

    # --- DPT reassemble: flatten nn.Sequential indices into named submodules ---
    # act_postprocess3 = [ProjectReadout, Transpose, Unflatten, Conv2d(1x1)]
    # act_postprocess4 = [ProjectReadout, Transpose, Unflatten, Conv2d(1x1), Conv2d(3x3 s2)]
    m = re.match(r"^backbone\.pretrained\.act_postprocess([34])\.(\d+)\.(.*)$", k)
    if m:
        level, index, rest = m.group(1), int(m.group(2)), m.group(3)
        if index == 0:
            # ProjectReadout.project = Sequential(Linear, GELU) -> "project.0.weight"
            rest = re.sub(r"^project\.0\.", "", rest)
            return f"backbone.pretrained.reassemble{level}.readout.project.{rest}"
        if index == 3:
            return f"backbone.pretrained.reassemble{level}.proj.{rest}"
        if index == 4:
            return f"backbone.pretrained.reassemble{level}.resize.{rest}"
        raise ValueError(f"unexpected act_postprocess index in {key}")

    # --- output head: Sequential(Conv2d, ReLU, MultitaskHead) ---
    m = re.match(r"^backbone\.scratch\.output_conv\.(\d+)\.(.*)$", k)
    if m:
        index, rest = int(m.group(1)), m.group(2)
        if index == 0:
            return f"backbone.scratch.output_conv.stem.{rest}"
        if index == 2:
            # heads.<i>.0 = Conv2d(3x3), heads.<i>.2 = Conv2d(1x1)
            mm = re.match(r"^heads\.(\d+)\.([02])\.(.*)$", rest)
            if mm:
                head, sub, tail = mm.group(1), mm.group(2), mm.group(3)
                name = "conv1" if sub == "0" else "conv2"
                return f"backbone.scratch.output_conv.heads.{head}.{name}.{tail}"
            raise ValueError(f"unexpected multitask head key {key}")
        raise ValueError(f"unexpected output_conv index in {key}")

    return k


def convert(checkpoint: Path, output_dir: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    use_layer_scale = any(k.endswith(".ls1.gamma") for k in state)
    upstream_params = sum(v.numel() for v in state.values() if v.dim() > 0)

    # ---- Fold conv+BatchNorm inside every ResidualConvUnit_custom ------------
    # Pattern: <unit>.conv{1,2}.weight (bias-free) followed by <unit>.bn{1,2}.*
    folded: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for key in list(state):
        m = re.match(r"^(.*)\.bn([12])\.weight$", key)
        if not m:
            continue
        unit, idx = m.group(1), m.group(2)
        conv_w_key = f"{unit}.conv{idx}.weight"
        if conv_w_key not in state:
            continue
        gamma = state[f"{unit}.bn{idx}.weight"].double()
        beta = state[f"{unit}.bn{idx}.bias"].double()
        mean = state[f"{unit}.bn{idx}.running_mean"].double()
        var = state[f"{unit}.bn{idx}.running_var"].double()
        scale = gamma / torch.sqrt(var + BN_EPS)

        conv_w = state[conv_w_key].double()
        folded[conv_w_key] = (conv_w * scale.reshape(-1, 1, 1, 1)).float()
        folded[f"{unit}.conv{idx}.bias"] = (beta - mean * scale).float()
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            consumed.add(f"{unit}.bn{idx}.{suffix}")

    # ---- Build the converted tensor set -------------------------------------
    out: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, value in state.items():
        if key in consumed:
            continue
        tensor = folded.get(key, value)

        new_key = remap_key(key)
        if new_key is None:
            dropped.append(key)
            continue

        if is_std_conv(key):
            tensor = standardize(tensor)

        if tensor.dim() == 4 and new_key.endswith(".weight"):
            tensor = to_nhwc(tensor)

        out[new_key] = tensor.contiguous().float()

    # Folded biases are new keys with no counterpart in `state`.
    for key, tensor in folded.items():
        if key.endswith(".bias"):
            new_key = remap_key(key)
            assert new_key is not None
            out[new_key] = tensor.contiguous().float()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(output_dir / "model.safetensors"))

    config = {
        "architecture": "scalelsd-vitbase",
        "use_layer_scale": use_layer_scale,
        "vit_features": 768,
        "vit_depth": 12,
        "vit_heads": 12,
        "patch_size": 16,
        "features": 256,
        "head_size": HEAD_SIZE,
        "stride": 2,
        "distance_threshold": 5.0,
        "source_checkpoint": checkpoint.name,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    converted_params = sum(v.numel() for v in out.values())
    print(f"source        : {checkpoint.name}")
    print(f"use_layer_scale: {use_layer_scale}")
    print(f"upstream keys : {len(state)}  ({upstream_params:,} params)")
    print(f"converted keys: {len(out)}  ({converted_params:,} params)")
    print(f"dropped       : {len(dropped)} ({', '.join(dropped[:4])}{'…' if len(dropped) > 4 else ''})")
    print(f"folded bn     : {sum(1 for k in folded if k.endswith('.bias'))} conv+bn pairs")
    print(f"written       : {output_dir / 'model.safetensors'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", "-c", required=True, type=Path)
    ap.add_argument("--output", "-o", required=True, type=Path)
    args = ap.parse_args()
    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
