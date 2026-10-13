# Dependencies & Reproduction Prerequisites

This file tells you **what is redistributable, what is not, and how to get the rest**.
Read it before you expect to hit the exact headline number (117.6 tokens/s on code).

---

## 1. What is and is not redistributable

| Asset | Origin | License | Redistributable? |
|---|---|---|---|
| SGLang v0.5.20 base | `sgl-project/sglang` | Apache-2.0 | **yes** |
| Community RDNA4 patch set (72 patches) | `mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference` | **no license file** | **no** — we do not redistribute it; you must obtain it from the author |
| **Our SGLang delta** (this repo, `patches/sglang/`) | this work | MIT (see `LICENSE`) | **yes** |
| `libr4d` kernel library | `StillDeadcode/libr4d` (Codeberg) | third-party | we publish **only our 20-line patch** (`patches/libr4d/`) |
| `radiance_mxfp4_fp8.so` decode GEMM | `paiton/radiance` (third-party) | **no source anywhere** | **no** — you extract it from paiton's public image |
| Model weights (Qwen3.8-27B AWQ + DFlash2 draft) | `z-lab` / checkpoint | **not redistributable** | **no** — we publish the **converter script only** (`reproduce/convert-awq-to-mxfp4.py`) |

---

## 2. The provenance chain (what your tree is built on)

Your `sgl-v0520` tree is, in order:

1. **SGLang v0.5.20** (upstream `94602c9`).
2. **The community RDNA4 patch set** — 72 patches from
   `mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference` (the 10-01 version, commit
   `784216d5c5`). This set does the heavy lifting: hipify of the `aot/csrc` kernels, the
   RDNA4-specific model fixes, the `mgehre` W4A16 MoE port, etc. **It has no license file,
   so we cannot redistribute it.** You must obtain it from the author and apply it yourself.
3. **Our delta** (this repo) — the MXFP4 quantization path, W4A16/W3 dispatch, GDN prefill
   integration, verify-attention glue, and the 20-line `libr4d` patch.

The reference files in `patches/sglang/reference/` are our **in-tree** versions of the mixed
files (files that both the community patch set and we modified). They are provided so you
can copy them over *if your base is the same 10-01 community patch set*. If your base
differs, use the `.patch` files in `patches/sglang/` (which apply cleanly to the 10-01 base)
and reconcile by hand.

---

## 3. How to get the non-redistributable pieces

### 3.1 The community RDNA4 patch set (72 patches)

Contact the author of `mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference`. The 10-01
version (commit `784216d5c5`) is the one our delta was cut against. Apply the 72 patches to
a pristine SGLang v0.5.20 tree, then apply our delta.

### 3.2 `radiance_mxfp4_fp8.so` (the decode GEMM)

This is the **third-party decode GEMM** that the main GEMM (27.77 ms/step, 66.6% of a step)
runs through. It is **not public as source** — it ships inside **paiton's public Docker
image**. To get it:

```bash
# pull paiton's public image (the exact tag is in the paiton project's docs)
docker pull <paiton-public-image>
docker create --name radiance-src <paiton-public-image>
docker cp radiance-src:/path/to/radiance_mxfp4_fp8.so ./
```

**Without this binary, the headline 117.6 is not reachable.** The fully-open "Open config"
(aiter/Triton W4A4) path measures **~24–32% lower TG**. This is the single biggest
reproducibility gap, and it is a licensing constraint, not an engineering one.

### 3.3 Model weights

We publish **only the converter** (`reproduce/convert-awq-to-mxfp4.py`). You must obtain:

- The **Qwen3.8-27B AWQ checkpoint** (the `qweight`/`qzeros`/`scales` tensors).
- The **DFlash2 draft** (`z-lab/Qwen3.8-27B-DFlash2`, 5-layer draft, chain topology).

Then convert the AWQ target to MXFP4 at load (the converter does this; see
`reproduce/README.md`).

---

## 4. Reproduction limits (what you *can* and *cannot* reproduce)

| Result | Reproducible with this repo alone? | Why |
|---|---|---|
| **The method and the tooling** (ruler, profiler route, attribution) | **yes** | fully open |
| **The findings** (roofline, per-kernel attribution, dead-ends) | **yes** | they are measurements, not binaries |
| **The patches** (our delta) | **yes** | MIT, apply to the 10-01 community base |
| **The 117.6 code TG** | **no, not without `radiance_mxfp4_fp8.so`** | the decode GEMM is third-party, not public as source |
| **The "Open config" TG** (~24–32% lower) | **yes** | aiter/Triton W4A4 is fully open |

**The honest summary:** everything *we* wrote is open and reproducible. The headline number
depends on a third-party decode GEMM that is not public as source. If you cannot get that
binary, you will land on the "Open config" path, which is ~24–32% lower in TG but is fully
open and reproducible.

---

## 5. The `radiance_mxfp4_fp8.so` interface (for the curious)

The decode GEMM is `radiance_mxfp4_fp8_gemm_decode<8,128,1,{1..5}>`, wired in
`patches/sglang/reference/.../r4d_mxfp4.py` with `RAD_DECODE=1`, `KS=8`, `NT=0`,
`EPIFAST=1`, `DEC_MAX_N=36864`. It is measured **24–32% faster** than the `libr4d` m64
kernel. The `RADIANCE_MXFP4_WPERM=1` env var enables the fragment-order weight permutation
that the kernel requires.

The `libr4d` m64 kernel (`gemm_bf16_nt_m16/m64`, `w4a16`, `w4a8`, `mxfp4a8`, all **M≤64**)
is the fallback for the decode band; it cannot carry prefill (a 2048-row prefill would
re-read the whole weight 32 times).
