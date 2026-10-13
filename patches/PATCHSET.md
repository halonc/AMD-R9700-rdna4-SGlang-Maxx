# Patch Set

This directory holds **our delta** — the SGLang patches and the `libr4d` patch that
implement the MXFP4 quantization path, the W4A16/W3 dispatch, the GDN prefill integration,
and the verify-attention glue.

---

## 1. The base you must have first

Our delta is cut against a specific base. **You must apply the community RDNA4 patch set
before applying our delta.**

| Layer | Value |
|---|---|
| SGLang base | v0.5.20 (upstream `94602c9`) |
| Community RDNA4 patch set | 72 patches from `mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference`, the **10-01 version, commit `784216d5c5`** |

The community patch set has **no license file**, so we do not redistribute it. Obtain it
from the author (see `../docs/DEPENDENCIES.md` §3.1).

---

## 2. Applying our delta

There are two ways to apply our delta, depending on how confident you are in your base:

### Option A — `git am` the `.patch` files (cleanest, applies to the exact 10-01 base)

```bash
cd sgl-v0520   # your SGLang v0.5.20 + 72 community patches tree
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/001-r4d-mxfp4-core.patch
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/002-r4d-w4.patch
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/003-r4d-w3.patch
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/004-r4d-gdn.patch
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/005-r4d-verify.patch
```

These five patches contain our **new files** (they are pure additions, so they apply
cleanly). The mixed files (files both the community set and we modified) are **not** in
these patches — see Option B.

### Option B — copy the reference files (robust to base drift)

The mixed files are provided as our **in-tree** versions in `reference/`. If your base is
the same 10-01 community patch set, you can copy them over directly:

```bash
cd sgl-v0520
cp -r ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/reference/python/sglang/srt/* \
      python/sglang/srt/
```

This overwrites: `awq_kernels.py`, `layernorm.py`, `models/qwen3_5.py`,
`quantization/awq/awq.py`, `quantization/fp8.py`, `quantization/__init__.py`,
`arg_groups/choices.py`, `configs/model_config.py`, and the `triton_3_6_0` / `triton_3_7_0`
MoE config JSONs. **If your base differs from the 10-01 community set, reconcile by hand**
(the reference files contain the community changes *plus* ours).

### The `libr4d` patch

```bash
cd libr4d   # the StillDeadcode/libr4d checkout
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/libr4d/001-sglang-kv-layout.patch
```

This 20-line patch adds a `const void* v` field to `R4DArgs` so the decode-attention kernel
can read SGLang's **separate K/V paged layout** (SGLang keeps K and V in separate buffers,
each `(num_blocks, block_size, kv_heads, head_dim)`, token-major inside a block, with no
K|V interleave — unlike the interleaved layout `libr4d` was written for).

---

## 3. What each patch does

### `sglang/001-r4d-mxfp4-core.patch`
- `r4d_mxfp4.py` — the MXFP4 core: E2M1 + E8M0 group-32 packing, the decode/prefill GEMM
  dispatch (`mxfp4_linear`, `mxfp4_linear_pq`, the chunked variants), the fragment-order
  weight permutation, the atiled fused-norm path, and the decode scratch.
- `r4d_mxfp4_config.py` — the `R4DMxfp4Config` quantization config.

### `sglang/002-r4d-w4.patch`
- `r4d_w4.py` — the W4A16/W4A8 dispatch: the `libr4d` WMMA fragment repack
  (`repack_cached`), the M≤64 decode GEMM (`gemm`), the W4A8 int8 prefill GEMM
  (`w4a8_gemm`), and the `MAX_M` / `KPB` / `GROUP` constants.

### `sglang/003-r4d-w3.patch`
- `r4d_w3.py` — the 3-bit (W3A4) ctypes binding to the `w3_decode.so` kernel: the
  SITES/TABLE dispatch, the merged-site repack, the gemv/chunked/prefill dispatch, and the
  rotation handling. (This line was **rejected** end-to-end — see `../docs/DEAD-ENDS.md`
  row 14 — but the binding is published for completeness.)

### `sglang/004-r4d-gdn.patch`
- `r4d_gdn.py` — the GDN prefill trio integration (`gdn_conv_prep` + `gdn_kkt_solve` +
  `gdn_chunk_scan`) for the ≥513-token whole-block GDN prefill.

### `sglang/005-r4d-verify.patch`
- `r4d_verify.py` — the SGLang-side verify-attention glue: it calls the `libr4d`
  verify-attention kernel with the separate K/V paged layout, handling the CUDA-graph
  host-side scalar freeze (the slot-index +1 offset and the `+t` snapshot index fix).

### `libr4d/001-sglang-kv-layout.patch`
- The 20-line `libr4d` patch (see §2 above).

---

## 4. The reference files (`reference/`)

`reference/` holds our **in-tree** versions of the **mixed files** — files that both the
community patch set and we modified. These are the "final working" versions from our
`sgl-v0520` tree. They are provided for two reasons:

1. **Copy-over convenience** — if your base is the 10-01 community set, copy them over
   (Option B above).
2. **Provenance clarity** — they show exactly what the final file looks like, so you can
   diff against the community base to see *our* contribution in context.

The mixed files are:

| File | Our contribution |
|---|---|
| `hardware_backend/gpu/quantization/awq_kernels.py` | the MXFP4 "path A" (convert once, free the AWQ copy, one forward path) + the `libr4d` W4A16 repack + the W3/W4 dispatch in `apply()` |
| `layers/layernorm.py` | the `fused_add_rms_quant` / `fused_add_rms_quant_tiled` MXFP4 fused-norm hooks |
| `models/qwen3_5.py` | the `mxfp4_ref` routing + the `_mxfp4_fused_norm_enabled` gate + the atiled prefill wiring |
| `layers/quantization/awq/awq.py` | the MXFP4 LM-head routing (the head is converted to MXFP4 and served by the decode GEMM) |
| `layers/quantization/fp8.py` | the DFlash2 draft fp8 → MXFP4 conversion at load |
| `layers/quantization/__init__.py` | registers `r4d_mxfp4` |
| `arg_groups/choices.py` | adds `r4d_mxfp4` to the quantization choices |
| `configs/model_config.py` | adds `r4d_mxfp4` to the supported-quant list |
| `moe/moe_runner/triton_utils/configs/triton_3_6_0/*.json` | the R9700 MoE Triton autotune configs (data, not code) |

> **Note:** the reference files contain the community changes *plus* ours. If your base
> differs from the 10-01 community set, do not copy them blindly — reconcile by hand.

---

## 5. The environment kill-switches

The patches read their tuning knobs from environment variables (not from SGLang's `Envs`
class). The production values are in `../../config/launch-qwen38-nas.sh`. The key ones:

| Env var | Production value | Effect |
|---|---|---|
| `SGLANG_MXFP4_ATILED_MIN_M` | **513** | atiled fused-norm for prefill ≥513 tokens |
| `SGLANG_R4D_GDN_MIN_T` | **513** | r4d GDN prefill trio for ≥513-token blocks |
| `SGLANG_R4D_ATTN` | **1** | enable the r4d verify-attention |
| `SGLANG_VERIFY_BLOCK_N` | **64** | verify block size (Triton fallback path only) |
| `RADIANCE_MXFP4_WPERM` | **1** | enable the fragment-order weight permutation |
| `SGLANG_EMBED_FP8` | **1** | fp8 embedding |
| `SGLANG_RDNA4_DISABLE_STORE_CACHE` | **1** | disable the store cache |
| `SGLANG_MXFP4_DRAFT` | **0** | MXFP4 draft conversion **off** (rejected — accept collapse) |
| `SGLANG_W3` | **0** | 3-bit W3A4 **off** (rejected end-to-end) |

**Fastest whole rollback**: set `SGLANG_MXFP4_ATILED_MIN_M` and `SGLANG_R4D_GDN_MIN_T` to 0
to return to the Phase 0 baseline.
