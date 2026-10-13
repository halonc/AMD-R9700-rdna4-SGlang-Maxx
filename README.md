# AMD R9700 (RDNA4) × SGLang — Qwen3.8-27B AWQ→MXFP4 + DFlash2

A set of **measured findings**, **kernels**, **patches**, and **tooling** for serving
**Qwen3.8-27B** (AWQ → MXFP4, with DFlash2 speculative decoding) on a single
**AMD R9700 32 GB** (gfx1201 / RDNA4) under **SGLang v0.5.20**.

The headline result: on code prompts, **tokens/s went from 76.4 (stock AWQ) to 117.6**
after the optimizations in this repo. All numbers are reproducible with the tooling here —
but a few third-party binaries are *not* redistributable, so read
[`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) before you expect to hit the exact headline.

---

## What this is

This is a **findings + code** repo, not a fork of SGLang. It contains:

1. **Measured findings** — the decode roofline, per-kernel attribution, the speed ladder,
   and a table of 12+ knobs that were tested and rejected. See [`docs/FINDINGS.md`](docs/FINDINGS.md)
   and [`docs/DEAD-ENDS.md`](docs/DEAD-ENDS.md).
2. **Our code** — SGLang patches (MXFP4 quantization path, W4A16/W3 dispatch, GDN prefill
   integration, verify-attention glue) plus a 20-line patch to the third-party `libr4d`
   kernel library. See [`patches/PATCHSET.md`](patches/PATCHSET.md).
3. **Measurement tooling** — a unified 3-pass benchmark ruler, a chrome-trace
   per-kernel aggregator, and the profiler route. See [`tooling/`](tooling/).
4. **Reproduce scripts** — the AWQ→MXFP4 weight converter, a `docker-compose.yml`, and
   the finalized launch configuration. See [`reproduce/`](reproduce/).

---

## The hardware / software stack

| Layer | Value |
|---|---|
| GPU | 1× AMD **R9700 32 GB** (gfx1201, RDNA4, 132 CU) |
| ROCm | 7.2.4 |
| PyTorch | 2.11.0+rocm7.2 |
| SGLang | **v0.5.20** (upstream `94602c9`) + community RDNA4 patch set + this repo's delta |
| Model | **Qwen3.8-27B** AWQ checkpoint, converted to **MXFP4** (E2M1 + E8M0 group-32) at load |
| Spec decode | **DFlash2** (5-layer draft, chain topology, `MAX_RUN=1`) |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` |

Model shape (from the checkpoint `config.json` → `text_config`): hidden 5120,
intermediate 17408, **64 layers** with `full_attention_interval: 4` ⇒ **16 full-attention
layers + 48 GDN (linear-attention) layers**, 24 Q heads / 4 KV heads, head_dim 256,
vocab 248320, GDN 48 value heads / 16 key heads / head_dim 128.

---

## The headline result (3-pass median, unified ruler)

`tokens/s` (TG) and `prefill tokens/s` (PP), code / Chinese prose / 20K technical Chinese:

| Stage | PP (4k / 8k / 20k) | TG (code / prose / 20K) | accept len (code / prose / 20K) |
|---|---|---|---|
| Phase 0 (stock AWQ, no MXFP4) | 2057.5 / 1847.6 / 1378.3 | — | — |
| + atiled fused-norm (MXFP4, `ATILED_MIN_M=513`) | 2256.6 / 2017.5 / 1474.3 | — | — |
| **+ r4d GDN prefill (`R4D_GDN_MIN_T=513`)** | **2353.5 / 2130.2 / 1529.9** | **117.6 / 32.6 / 57.9** | **4.83 / 1.506 / 3.08** |

The decode roofline is **631 GB/s** bandwidth, **14.40 GB/step** of weights, giving a
**22.8 ms/step hard floor**. The main GEMM runs at **27.77 ms/step (66.6% of a step)**,
i.e. **81% of the bandwidth roofline** — decode is bandwidth-bound, not compute-bound.
See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full attribution.

> **Reproducibility caveat:** the headline 117.6 requires the third-party
> `radiance_mxfp4_fp8.so` decode GEMM (not public as source). The fully-open "Open config"
> (aiter/Triton W4A4) path measures ~24–32% lower TG. Details in
> [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

---

## Directory layout

```
README.md               this file
LICENSE
docs/
  FINDINGS.md           measured results: roofline, per-kernel attribution, speed ladder
  DEAD-ENDS.md          12+ knobs tested and rejected, with the reason
  METHODOLOGY.md        how to measure: A/B discipline, 3-pass median, the ruler
  DEPENDENCIES.md       what is redistributable, what is not, and how to get the rest
patches/
  PATCHSET.md           provenance + how to apply the patches
  sglang/
    001..005-*.patch    our new files (MXFP4 core, W4A16, W3, GDN, verify)
    reference/          our in-tree versions of the mixed files (copy-over)
  libr4d/
    001-sglang-kv-layout.patch   20-line patch so decode attention reads SGLang's
                                 separate K/V paged layout
tooling/
  bench.py              unified 3-pass ruler (TG / PP / accept probe)
  trace_agg.py          chrome-trace → per-kernel attribution
  phase3_prof.sh        POST /start_profile → request → POST /stop_profile
  ba_kernel_test.py     Triton in_proj_ba kernel microbench
reproduce/
  convert-awq-to-mxfp4.py   AWQ → MXFP4 weight converter
  docker-compose.yml        SGLang service definition
  launch.env                finalized env vars + kill-switches
  README.md                 step-by-step reproduction
config/
  launch-qwen38-nas.sh      the production launch wrapper (env vars + flags)
```

---

## Provenance (what is whose)

This is important for licensing and for knowing what you can redistribute:

| Asset | Origin | Redistributable? |
|---|---|---|
| SGLang v0.5.20 base | `sgl-project/sglang` (Apache-2.0) | yes |
| Community RDNA4 patch set (72 patches) | `mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference` | **no license** — we do *not* redistribute it; you must obtain it separately |
| **Our SGLang delta** (this repo) | this work | yes (see `LICENSE`) |
| `libr4d` kernel library | `StillDeadcode/libr4d` (Codeberg) | third-party; we publish only our **20-line patch** |
| `radiance_mxfp4_fp8.so` decode GEMM | `paiton/radiance` (third-party) | **no source anywhere** — you extract it from paiton's public image |
| Model weights (Qwen3.8-27B AWQ + DFlash2 draft) | `z-lab` / checkpoint | **not redistributable** — we publish the **converter script only** |

---

## Quick start

```bash
# 1. Get the base: SGLang v0.5.20 + the community RDNA4 patch set (see DEPENDENCIES.md)
# 2. Apply our patches:
cd sgl-v0520
git am -p1 < ../AMD-R9700-rdna4-SGlang-Maxx/patches/sglang/001-r4d-mxfp4-core.patch
# ... (see patches/PATCHSET.md for the full order)
# 3. Convert your AWQ checkpoint to MXFP4:
python reproduce/convert-awq-to-mxfp4.py --in /path/to/awq --out /path/to/mxfp4
# 4. Launch (see reproduce/README.md for the full env + flags):
docker compose -f reproduce/docker-compose.yml up
```

Full step-by-step: [`reproduce/README.md`](reproduce/README.md).

---

## License

Our code and documentation in this repo are released under the **MIT License** (see
`LICENSE`). The third-party assets above keep their own terms; we do not redistribute them.
