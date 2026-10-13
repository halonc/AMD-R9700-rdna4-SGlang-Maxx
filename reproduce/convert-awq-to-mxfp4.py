#!/usr/bin/env python3
"""Convert a Qwen3.8-27B AWQ checkpoint to MXFP4 (E2M1 + E8M0 group-32).

This is the offline weight converter for the SGLang MXFP4 path. It reads the
AWQ ``qweight``/``qzeros``/``scales`` tensors, dequantizes to a dense fp32
weight, re-quantizes to MXFP4, and writes a new safetensors checkpoint with
``weight_packed`` (uint8 E2M1, [N, K/2]) and ``weight_scale`` (uint8 E8M0,
[K/32, N]) tensors.

The conversion is the same one the SGLang runtime does at load
(``awq_kernels.py::_mx_build`` + ``nvfp4_to_mxfp4.to_mxfp4``); doing it offline
means the runtime reads MXFP4 directly and never keeps the AWQ copy resident.

Usage:
    python convert-awq-to-mxfp4.py --in /path/to/awq --out /path/to/mxfp4

Notes:
    * Only Linear layers (``qweight``/``qzeros``/``scales``) are converted.
      Everything else (embeddings, norms, the GDN ``in_proj_a/b`` bf16
      full-precision modules, etc.) is copied through unchanged.
    * The LM head is converted too (it is served by the decode GEMM).
    * The output is byte-identical to what the runtime would produce at load.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors import save_file

# ---------------------------------------------------------------------------
# E2M1 / E8M0 quantization (from nvfp4_to_mxfp4.py)
# ---------------------------------------------------------------------------
E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
BLK = 32
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def quantise_e2m1(q: torch.Tensor) -> torch.Tensor:
    """q normalised to the E2M1 range -> uint8 codes (0..7, bit3 = sign)."""
    idx = torch.bucketize(q.abs(), _E2M1_MID.to(q.device))
    return idx.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)


def to_mxfp4(w_fp32: torch.Tensor):
    """fp32 [N, K] -> (packed [N, K/2] uint8, e8m0 [K/32, N] uint8)."""
    N, K = w_fp32.shape
    assert K % BLK == 0
    g = w_fp32.view(N, K // BLK, BLK)
    amax = g.abs().amax(dim=-1)                                   # [N, nb]
    e = torch.ceil(torch.log2((amax / E2M1_MAX).clamp(min=2 ** -127)))
    e = e.clamp(-127, 126)
    sc = torch.pow(2.0, e)                                        # [N, nb]
    q = g / sc[..., None]
    code = quantise_e2m1(q).view(N, K // 2, 2)                    # low = even k
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8).contiguous()
    e8m0 = (e + 127).clamp(0, 254).to(torch.uint8).t().contiguous()
    return packed, e8m0


def dequant_mxfp4(packed, e8m0, K):
    """packed [N, K/2] u8 + e8m0 [K/32, N] u8 -> fp32 [N, K] (for the self-check)."""
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    v_lo = E2M1_POS[lo & 7] * torch.where(((lo >> 3) & 1) == 1, -1.0, 1.0)
    v_hi = E2M1_POS[hi & 7] * torch.where(((hi >> 3) & 1) == 1, -1.0, 1.0)
    w = torch.stack([v_lo, v_hi], dim=-1).flatten(-2)
    s = (2.0 ** (e8m0.t().float() - 127)).repeat_interleave(BLK, dim=1)[:, :K]
    return w * s


# ---------------------------------------------------------------------------
# AWQ interleave + dequantization (from awq_kernels.py)
# ---------------------------------------------------------------------------
# AWQ packs 8 output columns per int32 in the order 0,2,4,6,1,3,5,7. Reading
# the nibbles sequentially permutes the weight matrix, so we must apply this
# interleave to recover the true column order.
_AWQ_INTERLEAVE = (0, 4, 1, 5, 2, 6, 3, 7)


def _mx_unpack4(p):
    """[K, N/8] int32 -> [K, N] codes, honouring the AWQ interleave."""
    K, N8 = p.shape
    b = p.view(torch.uint8).view(K, N8, 4)
    seq = torch.stack([b & 0x0F, (b >> 4) & 0x0F], dim=-1).reshape(K, N8, 8)
    return seq[..., list(_AWQ_INTERLEAVE)].reshape(K, N8 * 8)


def awq_dequantize(qweight, qzeros, scales):
    """AWQ (qweight, qzeros, scales) -> dense fp32 [K, N].

    qweight [K, N/8] int32, qzeros [K, N/8] int32, scales [K/128, N] (group-128).
    """
    w_q = _mx_unpack4(qweight).to(torch.float32)   # [K, N] 4-bit codes
    z_q = _mx_unpack4(qzeros).to(torch.float32)     # [K, N] 4-bit codes
    K, N = w_q.shape
    g = scales.shape[0]
    assert K % g == 0, f"K={K} not divisible by group={g}"
    # dequantize: (code - zero) * scale, per group of 128
    dense = ((w_q.view(g, 128, N) - z_q[:, None, :])
             * scales.to(torch.float32)[:, None, :]).view(K, N)
    return dense.t().contiguous()                  # [N, K]


# ---------------------------------------------------------------------------
# Fragment order (from r4d_mxfp4.py)
# ---------------------------------------------------------------------------
def fragment_order(packed: torch.Tensor, K: int) -> torch.Tensor:
    """[N, K/2] uint8 checkpoint order -> WMMA fragment order (same shape)."""
    N = packed.shape[0]
    nt, ks = N // 16, K // 16
    return (packed.view(nt, 16, ks, 2, 4)
                  .permute(0, 2, 3, 1, 4)
                  .contiguous()
                  .view(N, K // 2))


# ---------------------------------------------------------------------------
# Checkpoint conversion
# ---------------------------------------------------------------------------
def is_awq_linear(key: str, tensors: dict) -> bool:
    """True if the key stem has the AWQ Linear trio (qweight/qzeros/scales)."""
    for suf in ("qweight", "qzeros", "scales"):
        if f"{key}.{suf}" not in tensors:
            return False
    return True


def convert(in_dir: str, out_dir: str, wperm: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    # load the index
    index_path = os.path.join(in_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        idx = json.load(open(index_path))["weight_map"]
    else:
        # single-file checkpoint
        idx = {}
        for fn in os.listdir(in_dir):
            if fn.endswith(".safetensors"):
                with safe_open(os.path.join(in_dir, fn), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        idx[k] = fn

    out_tensors = {}
    n_converted = 0
    n_copied = 0

    # group keys by their file
    by_file = {}
    for k, fn in idx.items():
        by_file.setdefault(fn, []).append(k)

    for fn, keys in by_file.items():
        with safe_open(os.path.join(in_dir, fn), framework="pt", device="cpu") as f:
            for k in keys:
                if k.endswith(".qweight"):
                    stem = k[:-len(".qweight")]
                    if is_awq_linear(stem, idx):
                        qw = f.get_tensor(stem + ".qweight")
                        qz = f.get_tensor(stem + ".qzeros")
                        sc = f.get_tensor(stem + ".scales")
                        dense = awq_dequantize(qw, qz, sc)        # [N, K] fp32
                        N, K = dense.shape
                        pk, e8 = to_mxfp4(dense)                 # [N,K/2] u8, [K/32,N] u8
                        if wperm and N % 16 == 0 and K % 16 == 0:
                            pk = fragment_order(pk, K)
                        out_tensors[stem + ".weight_packed"] = pk
                        out_tensors[stem + ".weight_scale"] = e8
                        n_converted += 1
                        # mark the AWQ trio as consumed (do not copy through)
                        for suf in ("qweight", "qzeros", "scales"):
                            idx.pop(stem + "." + suf, None)
                        continue
                # copy everything else through unchanged
                out_tensors[k] = f.get_tensor(k)
                n_copied += 1

    # write out (single file for simplicity; split for very large checkpoints)
    save_file(out_tensors, os.path.join(out_dir, "model.safetensors"))
    # write a minimal index
    index = {"weight_map": {k: "model.safetensors" for k in out_tensors}}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh)

    # copy config.json and any non-tensor files
    for fn in os.listdir(in_dir):
        if fn.endswith((".json", ".txt")) and fn != "model.safetensors.index.json":
            with open(os.path.join(in_dir, fn)) as src:
                with open(os.path.join(out_dir, fn), "w") as dst:
                    dst.write(src.read())

    print(f"converted {n_converted} AWQ Linear layers -> MXFP4")
    print(f"copied {n_copied} other tensors through unchanged")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="AWQ checkpoint dir")
    ap.add_argument("--out", dest="out_dir", required=True, help="MXFP4 output dir")
    ap.add_argument("--no-wperm", action="store_true",
                    help="skip the fragment-order permutation (disable RADIANCE_MXFP4_WPERM)")
    args = ap.parse_args()
    convert(args.in_dir, args.out_dir, wperm=not args.no_wperm)
