#!/usr/bin/env python3
"""Dump per-stage PyTorch activations so the Swift port can be bisected against them.

    PYTHONPATH=<scalelsd repo> python Scripts/make_fixtures.py \
        --checkpoint scalelsd-vitbase-v1-train-sa1b.pt \
        --image assets/indoor.jpg --output Tests/MLXScaleLSDTests/Fixtures/v1

Every spatial tensor is stored NHWC so the Swift side can compare without transposing;
token tensors stay (B, N, C). The first stage whose relative error jumps identifies the
broken layer — see docs/PARITY.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

from scalelsd.ssl.misc.train_utils import load_scalelsd_model


def nhwc(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W, C)."""
    return t.permute(0, 2, 3, 1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", "-c", required=True, type=Path)
    ap.add_argument("--image", "-i", required=True, type=Path)
    ap.add_argument("--output", "-o", required=True, type=Path)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    model = load_scalelsd_model(str(args.checkpoint), device="cpu")
    model.eval()

    gray = cv2.imread(str(args.image), 0)
    resized = cv2.resize(gray, (args.size, args.size))
    image = torch.from_numpy(resized).float() / 255.0
    image = image[None, None]  # (1, 1, H, W)

    out: dict[str, torch.Tensor] = {"input": nhwc(image)}

    pretrained = model.backbone.pretrained
    scratch = model.backbone.scratch
    vit = pretrained.model

    handles = []
    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_module, _inputs, output):
            captured[name] = output if isinstance(output, torch.Tensor) else output[0]

        return hook

    # ResNet taps (DPT levels 1 and 2) and the two ViT taps (levels 3 and 4).
    handles.append(vit.patch_embed.backbone.stem.register_forward_hook(grab("stem")))
    handles.append(vit.patch_embed.backbone.stages[0].register_forward_hook(grab("stage0")))
    handles.append(vit.patch_embed.backbone.stages[1].register_forward_hook(grab("stage1")))
    handles.append(vit.patch_embed.backbone.stages[2].register_forward_hook(grab("stage2")))
    handles.append(vit.patch_embed.proj.register_forward_hook(grab("patch_proj")))
    handles.append(vit.blocks[0].register_forward_hook(grab("block0")))
    handles.append(vit.blocks[8].register_forward_hook(grab("tap8")))
    handles.append(vit.blocks[11].register_forward_hook(grab("tap11")))
    def grab_input(name):
        """Capture a module's *input* — used for the reassembled DPT levels, which are
        exactly what feeds `layerN_rn`."""

        def hook(_module, inputs):
            captured[name] = inputs[0]

        return hook

    handles.append(scratch.layer3_rn.register_forward_pre_hook(grab_input("layer3")))
    handles.append(scratch.layer4_rn.register_forward_pre_hook(grab_input("layer4")))
    handles.append(scratch.layer1_rn.register_forward_hook(grab("layer1_rn")))
    handles.append(scratch.layer2_rn.register_forward_hook(grab("layer2_rn")))
    handles.append(scratch.layer3_rn.register_forward_hook(grab("layer3_rn")))
    handles.append(scratch.layer4_rn.register_forward_hook(grab("layer4_rn")))
    handles.append(scratch.refinenet4.register_forward_hook(grab("path4")))
    handles.append(scratch.refinenet3.register_forward_hook(grab("path3")))
    handles.append(scratch.refinenet2.register_forward_hook(grab("path2")))
    handles.append(scratch.refinenet1.register_forward_hook(grab("path1")))

    with torch.no_grad():
        field, _, _ = model.forward_backbone(image)

    for handle in handles:
        handle.remove()

    for name, tensor in captured.items():
        out[name] = nhwc(tensor) if tensor.dim() == 4 else tensor.contiguous()
    out["output"] = nhwc(field)

    # ---- end-to-end detections, the metric that actually matters -------------
    from scalelsd.ssl.models.detector import ScaleLSD as _S

    _S.num_junctions_inference = 512
    _S.junction_threshold_hm = 0.008
    meta = {
        "width": gray.shape[1],
        "height": gray.shape[0],
        "filename": "",
        "use_lsd": False,
        "use_nms": False,
    }
    with torch.no_grad():
        results, _ = model(image, meta)
    result = results[0]
    out["image_size"] = torch.tensor([gray.shape[1], gray.shape[0]], dtype=torch.float32)
    out["lines_pred"] = result["lines_pred"].contiguous()
    out["lines_score"] = result["lines_score"].contiguous()
    out["juncs_pred"] = result["juncs_pred"].contiguous()
    out["juncs_score"] = result["juncs_score"].contiguous()

    # ---- LSD-rectifier fixtures ---------------------------------------------
    # `scalelsd.base.csrc._C.encodels` is a CUDA extension and cannot run here, so the
    # reference for the direction field is transcribed scalar-for-scalar from
    # scalelsd/base/csrc/linesegment.cu (encode_kernel) below.
    lsd = cv2.createLineSegmentDetector(0)  # 0 == LSD_REFINE_NONE
    detected = lsd.detect(resized)[0]
    if detected is not None:
        lsd_lines = detected.reshape(-1, 4).astype(np.float32)
        stride = 2
        grid_h, grid_w = args.size // stride, args.size // stride
        scaled = lsd_lines / stride

        # encode_kernel: nearest point on the nearest segment, per pixel.
        ys, xs = np.meshgrid(
            np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        px = xs.reshape(-1, 1)
        py = ys.reshape(-1, 1)
        x1, y1, x2, y2 = (scaled[:, i][None, :] for i in range(4))
        dx, dy = x2 - x1, y2 - y1
        norm2 = dx * dx + dy * dy
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / (norm2 + 1e-6), 0.0, 1.0)
        ax = x1 + t * dx - px
        ay = y1 + t * dy - py
        best = np.argmin(ax * ax + ay * ay, axis=1)
        rows = np.arange(px.shape[0])
        md_angle = np.arctan2(ay[rows, best], ax[rows, best])
        md_angle_n = md_angle / (2 * np.pi) + 0.5

        out["lsd_lines"] = torch.from_numpy(lsd_lines)
        out["lsd_direction"] = torch.from_numpy(
            md_angle_n.reshape(1, grid_h, grid_w, 1).astype(np.float32)
        )
        print(f"  (lsd: {lsd_lines.shape[0]} segments)")

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.float().contiguous() for k, v in out.items()},
        str(args.output / "fixtures.safetensors"),
    )

    print(f"wrote {args.output / 'fixtures.safetensors'}")
    for name, tensor in out.items():
        print(f"  {name:12s} {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
