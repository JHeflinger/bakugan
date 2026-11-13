#!/usr/bin/env python3
"""Exports DINOv3 as the generator's semantic loss: one ONNX graph that takes the design's views and the target views,
and gives each view's loss and the gradient of the summed loss with respect to every pixel of both.

    pip install torch transformers onnx onnxscript onnxruntime
    hf auth login                    # (or huggingface-cli login) the DINOv3 weights are gated: accept their license on the model page first
    python3 export_dinov3.py                                   # ViT-S/16 at 112x112 -> dinov3_loss_112.onnx
    python3 export_dinov3.py --size 224                        # finer: 14x14 patches instead of 7x7, ~4x slower
    python3 export_dinov3.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
    python3 export_dinov3.py --random                          # random weights, for testing the plumbing offline

The graph compares one view at a time, at the size it was exported for (S, a multiple of the patch size, 16); the
generator reads S from the file and renders its views at that size:
    render       float [1, 1, S, S]   a view of the design, grey, 0..1
    target       float [1, 1, S, S]   what it should look like (the mesh, as placed), grey, 0..1
    patch_weight float [1]            0: compare the global (CLS) embeddings, 1: the patch embeddings, between: a mix
  ->
    loss         float [1]            (1 - w) (1 - cos(global_r, global_t)) + w mean_p (1 - cos(patch_r, patch_t))
    grad_render  float [1, 1, S, S]   d loss / d render
    grad_target  float [1, 1, S, S]   d loss / d target (the mesh's placement moves it)

Patches are compared position by position: the two views share a camera, so patch p of one covers what patch p of the
other does. Grey goes in as R = G = B; the ImageNet normalization is folded into the patch embedding.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import transformers.models.dinov3_vit.modeling_dinov3_vit as dm
from transformers import DINOv3ViTConfig, DINOv3ViTModel

# its cache can't hash the symbolic sizes used while tracing
dm.get_patches_center_coordinates = getattr(dm.get_patches_center_coordinates, "__wrapped__", dm.get_patches_center_coordinates)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class GreyPatches(torch.nn.Module):
    """The patch embedding (a stride-16 convolution) for grey input, with ImageNet's normalization folded in, written as
    a reshape and a matrix product. Same result; its gradient exports to ONNX, which the convolution's doesn't."""

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        w = conv.weight.detach()                                   # [C, 3, P, P]
        std = torch.tensor(STD).view(1, 3, 1, 1)
        mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        self.p = conv.kernel_size[0]
        self.register_buffer("weight", (w / std).sum(1).flatten(1).t().contiguous())  # [P*P, C]
        bias = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
        self.register_buffer("b", bias - (w * mean / std).sum((1, 2, 3)))

    def forward(self, x):                                          # [N, 1, H, W] -> [N, C, H/P, W/P]
        n, _, h, wd = x.shape
        p = self.p
        x = x.reshape(n, h // p, p, wd // p, p).permute(0, 1, 3, 2, 4).reshape(n, (h // p) * (wd // p), p * p)
        y = x @ self.weight + self.b
        return y.transpose(1, 2).reshape(n, -1, h // p, wd // p)


def load(args):
    if args.random:
        torch.manual_seed(0)
        cfg = DINOv3ViTConfig(num_register_tokens=4, num_hidden_layers=args.layers)
        cfg._attn_implementation = "eager"
        model = DINOv3ViTModel(cfg)
    else:
        model = DINOv3ViTModel.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    model.embeddings.patch_embeddings = GreyPatches(model.embeddings.patch_embeddings)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m", help="Hugging Face id or local folder")
    ap.add_argument("--size", type=int, default=112, help="the views' size in pixels (a multiple of 16)")
    ap.add_argument("--out", default=None, help="default: dinov3_loss_<size>.onnx")
    ap.add_argument("--random", action="store_true", help="random weights (no download), to test the plumbing")
    ap.add_argument("--layers", type=int, default=12, help="with --random: how many transformer layers")
    args = ap.parse_args()
    if args.size % 16:
        sys.exit("--size must be a multiple of 16")
    args.out = args.out or f"dinov3_loss_{args.size}.onnx"

    model = load(args)
    registers = model.config.num_register_tokens
    names = [k for k, _ in model.named_parameters()] + [k for k, _ in model.named_buffers()]
    consts = [v for _, v in model.named_parameters()] + [v for _, v in model.named_buffers()]

    def loss(render, target, weight, *ps):
        n = render.shape[0]
        out = torch.func.functional_call(model, dict(zip(names, ps)), (), {"pixel_values": torch.cat([render, target])})
        tokens = out.last_hidden_state                             # [2N, 1 + registers + patches, C], normed
        cls, patches = tokens[:, 0], tokens[:, 1 + registers:]
        g = 1.0 - torch.nn.functional.cosine_similarity(cls[:n], cls[n:], dim=-1, eps=1e-6)
        p = (1.0 - torch.nn.functional.cosine_similarity(patches[:n], patches[n:], dim=-1, eps=1e-6)).mean(1)
        w = weight.reshape(())
        per = (1.0 - w) * g + w * p
        return per.sum(), per

    def graph(render, target, weight, *ps):
        (gr, gt), (_, per) = torch.func.grad_and_value(loss, argnums=(0, 1), has_aux=True)(render, target, weight, *ps)
        return per, gr, gt

    # (traced at one size: symbolic sizes come out baked in somewhere in the backward pass anyway)
    x = torch.rand(1, 1, args.size, args.size)
    t = torch.rand(1, 1, args.size, args.size)
    w = torch.tensor([0.5])
    print("tracing the loss and its gradient...")
    gm = make_fx(graph)(x, t, w, *consts)

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i, c in enumerate(consts):
                self.register_buffer(f"c{i}", c.detach().clone())

        def forward(self, render, target, patch_weight):
            return gm(render, target, patch_weight, *[getattr(self, f"c{i}") for i in range(len(consts))])

    print("exporting...")
    torch.onnx.export(Graph(), (x, t, w), args.out, dynamo=True, external_data=False,
                      input_names=["render", "target", "patch_weight"], output_names=["loss", "grad_render", "grad_target"])

    # check it against PyTorch
    import onnxruntime as ort
    s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    shape = (1, 1, args.size, args.size)
    for weight in (0.0, 0.3, 1.0):
        a = np.random.rand(*shape).astype(np.float32)
        b = np.random.rand(*shape).astype(np.float32)
        wt = np.array([weight], np.float32)
        o = s.run(None, {"render": a, "target": b, "patch_weight": wt})
        ref = graph(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(wt), *consts)
        err = max(float(np.abs(o[k] - ref[k].numpy()).max() / (np.abs(ref[k].numpy()).max() + 1e-12)) for k in range(3))
        print(f"  {shape} patch_weight {weight}: loss {o[0]}, worst relative difference from PyTorch {err:.1e}")
        if err > 1e-3:
            sys.exit("the exported graph doesn't match PyTorch")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
