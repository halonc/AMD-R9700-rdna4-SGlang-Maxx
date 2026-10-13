# Dead Ends — Knobs Tested and Rejected

This is the table of knobs that were **tested and rejected**, with the measured reason.
The point of this file is to save you the GPU time: **do not retry these**. The recurring
law that shows up three times (libr4d GDN verify, 3-bit, fused silu) is:

> **Any optimization that changes the target's numeric shape — measure the accept rate
> before you measure the kernel time.** A faster kernel that flips the greedy-accept
> decision is a net loss.

All numbers are 3-pass medians unless noted. "TG" = tokens/s, "PP" = prefill tokens/s.

---

| # | Knob | Expected | Measured | Verdict |
|---|---|---|---|---|
| 1 | `--speculative-dflash-block-size 4/6` | verify fewer tokens | M-independent (§FINDINGS.4); accept length drops | **rejected** |
| 2 | **DSPARK** (`--speculative-algorithm DSPARK` + `RadixArk/Qwen3.8-27B-DSpark`) | Markov head raises accept + ragged verify saves time | accept code 4.65 / prose **1.67** / 20K **2.75**; TG 89.2 / 33.3 / 33.3; step 52.2 / 50.2 / **82.7 ms** | **net loss** — the Markov head does raise accept (prose +11%, 20K +10%), but that draft has **no sliding window** (`use_sliding_window: False`) ⇒ draft attention O(ctx), 20K step +56%; and the 1.86 B draft shrinks the KV pool 178,528 → 111,824 |
| 3 | `--speculative-draft-window-size 2048` | give the draft a sliding window to save 20K time | accept histogram **bit-identical** to production; TG code 97.8/100.9, prose 32.2/32.7, 20K 47.0 (production 100.7/32.8/47.4) | **no gain** — 131K saves only ~2 ms/step (~3%), not worth the config surface |
| 4 | `MAMBA 8` | save 1.15 GB, KV 178,528 → 207,168 | from the 4th follow-up prompt, 0.2 → 3.3 s | **rejected**, keep 16 |
| 5 | draft fp8→MXFP4 (`SGLANG_MXFP4_DRAFT=1`) | save 1.5 ms/step | accept rate 0.68 → 0.25 | **rejected**, keep 0 |
| 6 | `DPM=high` | boost clocks | 4 ms/step slower (hits the power wall) | keep `auto` |
| 7 | `CHUNK=4096` | faster prefill | FLA era: `fla/chunk_o.py:148` OOM crash (hit twice). **Re-tested 2026-10-07 under the r4d GDN prefill path: no OOM, 3 passes exit cleanly**, but PP only +0.1~2.5%, TG −1.2~1.6%, free VRAM drops to 0.73–0.81 GiB | keep **2048**; the rejection reason changed from "OOM" to "measured no net gain" |
| 8 | **libr4d GDN decode / verify pair** (`gdn_conv_update_w4_h128_bf16` + `gdn_recurrent_update_k128_v128_bf16_fp32state`) | eat the decode gap (SGLang 41.2 vs paiton 29.5 ms/step) | ① `forward_decode` is **never called** under DFlash2 (`speculative_attention_mode=prefill`, every step's target is the extend's `is_target_verify`; wiring the decode branch and running a full round gives diagnostic count 0). ② Re-wired to verify (DFlash2 candidates are a **chain**: `retrieve_next_sibling` all −1) end-to-end: TG 117.6 → 107.3 / 32.6 → 29.2 / 57.9 → **47.4**, PP unchanged, accept 4.83 → 4.063 / 1.506 → 1.369 / 3.08 → 2.667. Microbench is **3.2×** (77 vs 247 µs/layer/step), unit parity all pass (out rel 0.36%, snapshot rel 0.09%, conv bit-identical) | **rejected**: `SGLANG_R4D_GDN_DECODE_MIN_T` / `SGLANG_R4D_GDN_VERIFY_MIN_T` default 0. The drop is not a slow kernel — it is that **GDN verify output goes straight into the greedy-accept decision**, and a 0.36% bf16-level difference flips the marginal decision, accept drops 9–16% |
| 9 | `SGLANG_SIMULATE_ACC_LEN` to calibrate M | sweep M without restart | does not change verify M (§FINDINGS.4) | **ineffective** |
| 10 | `SGLANG_ALT_STREAM=1` + `SGLANG_GDN_QKVZ_BA_ALT_STREAM=1` (put GDN's 49 tiny bf16 `in_proj_ba` GEMMs per layer on an alt stream to overlap with the main GEMM) | hide the 1.52 ms/step Tensile `Cijk…MT16x16x32` in the profile | TG code **78.5** / prose 22.4 / 20K **41.3** (production 117.6 / 32.6 / 57.9, **−33%**); PP 2345/2125/1529 and accept 4.83/1.506/3.08 **completely unchanged** | **rejected**: on ROCm, decode-stage dual-stream is a net loss (cross-stream event sync stretches each step by ~1/3). `radiance_mxfp4_fp8_gemm_decode` KS/NT also swept (KS=4/8/16, NT=0/1, 6 production shapes + multi-copy weight-rotation microbench): 24.43–25.01 ms/step, **completely flat** ⇒ GEMM config already at local optimum |
| 11 | radiance fused `launch_silu_mul_quant` (= vLLM `HOIST_QUANT` equivalent; switch `SGLANG_MXFP4_FUSED_SILU`; code fully rolled back after the 2026-10-09 test) | kill the MLP tail `act_and_mul` + `_quant_act_fp8_kernel` (profile 0.29 + 1.31 ms/step) | offline **CUDA-graph replay** (`silu_graph_bench.py`): M=9 20.00 → **7.30 µs (−63.5%)**, M=64 31.17 → 11.10 µs ⇒ −0.81 ~ −1.28 ms/step. End-to-end 3 pass: TG 117.6 → **113.2** / 32.6 → 33.5 / 57.9 → **56.2**, PP +0.4~1.2% (noise, fusion only acts on M≤64), accept 4.83 → **4.414 (−8.6%)** | **rejected**, code fully rolled back (2026-10-09: three source files and the test byte-identical to pre-patch md5, stale pyc cleared, no dead switch left; the patched version kept as `*.patched-20261009-silu` for reference). Reason: different quantization convention (q byte agreement 96–98%, **scale agreement 0–35%**, radiance slightly coarser), and the fusion only changes the target (the fp8 draft has no `mxfp4_w`, does not go through the fusion) ⇒ target/draft divergence grows ⇒ greedy accept drops. **The same law hits a third time** (libr4d GDN verify, 3-bit, fused silu) |
| 12 | radiance's other GEMM knobs `DECODE_BK` / `TN4_MIN_M` / `PADOUT` / `PERBLOCK_NK` (2026-10-09) | maybe 1–2 ms still hidden | per-step GEMM: baseline re-test 25.27 / 24.78 (**noise band ±0.5**); `DECODE_BK`=32/64/128/256 → 25.33 / **39.24 (bad)** / 24.79 / 25.00 (128 re-test 24.63); `TN4_MIN_M`=0/128 → 25.19 / 25.22; `PADOUT=1` 24.68; `PERBLOCK_NK=5120:17408` 25.06 | **all within noise, no new lever** — the "GEMM already at local optimum" claim was earned only after sweeping the whole library, not just KS/NT |
| 13 | `CHUNK=8192` (the external post's parameter, 2026-10-09 re-test) | faster prefill | MEMF 0.94 → 20K prefill `OutOfMemoryError: Tried to allocate 80.00 MiB … 0 bytes free` (already 30.76 GiB used); MEMF 0.90 → still OOM (needs 544 MiB, only 130 MiB left) | same row as `CHUNK=4096`, rejected; new fact: **vLLM can run 8192 because it counts peak transient memory into the KV budget; SGLang's `mem_fraction_static` locks KV+mamba first** ⇒ the external post's parameter set is not portable to SGLang's memory model |
| 14 | **3-bit W3A4** (whole line) | byte-count win (3k/8 vs 4k/8) | microbench is 25–79% faster per GEMM, but end-to-end: step 44.7 → **46.3 ms** (up, not down), accept 4.83 → **2.753** (collapsed); FUSED_NORM=0 leaves accept bit-identical while TG drops to 92.9 | **rejected** — see `FINDINGS.md` §6; the byte-count win cannot survive in-situ overhead + the accept collapse |
| 15 | hand-written Triton kernel to replace `in_proj_ba` (`x[9,5120]@W[48,5120]ᵀ`, offline tested, **not wired into the service**) | save the 1.52 ms/step Tensile kernel | M=1/9/16/17 **bit-identical** to `F.linear`; M=9: 35.7 µs → **21.5 µs** (`tooling/ba_kernel_test.py`) | saves only **0.68 ms/step (+1.5% TG)**, not 1.5 ms (do not count all of Tensile as savable). 21.5 µs is still slow (491 KB of weights should be 1–2 µs of bandwidth time); press it below 10 µs before wiring |
| 16 | "port the r4d kernels into paiton" (plan A) | speed up the production 8080 | the paiton container's `r4d.so` (a 9-18 build) and `radiance_mxfp4_fp8.so` are **already loaded** in the running EngineCore, and that build **already has** `gdn_conv_prep` / `gdn_kkt_solve` / `gdn_chunk_scan` / `attn_prefill_h256_gqa6_*`; paiton also has `v1/attention/backends/gdn_attn.py` referencing them | **gain ≈ 0**: the +15% PP SGLang won on 2026-10-07 is essentially putting paiton's long-existing kernels into SGLang. **Do not overwrite the local r4d.so into the container** — the two have diverged, and the container build's unique `gdn_fused_update_w4k128v128_bf16*` / `gdn_lazy_update_*` / `gdn_recurrent_update_*_{bf16,f16}state` are exactly the entry points paiton uses; overwriting will crash it |

---

## The recurring law

Three independent optimizations (libr4d GDN verify, 3-bit, fused silu) all lost end-to-end
for the same reason: **they changed the target's numeric shape, which flipped the
greedy-accept decision, and the accept-rate collapse more than offset the kernel-time win.**

Before you try any "faster kernel" change, run the accept-rate probe
(`tooling/bench.py` prints `accept_probe=`) on the same prompt set. If accept drops, the
change is a net loss regardless of how fast the kernel is.
