# Measured Findings

All numbers below are **3-pass medians** from the unified ruler
(`tooling/bench.py`) unless noted otherwise. The ruler: same prompt set, streaming,
`temperature=0`, `chat_template_kwargs:{enable_thinking:false}`, `max_tokens=256`,
`ignore_eos`, 3 passes, median. **Never re-run the full GSM8K-300 set** — it is a
quality gate, not a speed gate, and it is not part of the speed ladder.

---

## 1. The decode roofline

Single-stream decode reads **all** weights from VRAM every step. The per-step weight
traffic is **14.40 GB** (not 12.6 — the early figure missed the 0.675 GB `lm_head`):

| Block | Per layer | Layers | Total |
|---|---|---|---|
| `mlp.gate_up` 34816×5120 | 272 MB | 64 | 8.70 GB |
| `mlp.down` 5120×17408 | 136 MB | 64 | 4.35 GB |
| `gdn.in_proj` 24576×5120 | 192 MB | 48 | 2.30 GB |
| `lm_head` 31040×5120 × 8 chunks | — | 1 | 0.675 GB |

The R9700 measures **547 GB/s copy / 631 GB/s read**, so the **hard floor is
14.40 / 631 = 22.8 ms/step**. Any kernel work should be judged against this number first.

**The main GEMM runs at 27.77–28.12 ms/step ⇒ 81% of the bandwidth roofline.** Decode is
bandwidth-bound, not compute-bound. The only way to go faster is to read *fewer* bytes
(better quantization / sparsity), not to optimize the GEMM kernel.

---

## 2. Per-kernel attribution (graph mode, code prompt)

From a 263-step decode window (step count derived from the kernels that fire exactly
once per step), `step ≈ 45.7 ms`, **GPU busy 42.2 ms/step, decode occupancy 92%**,
median inter-kernel gap **4.3 µs** ⇒ the ~15 ms of non-GEMM time is *not* launch/scheduling
overhead and *not* "unattributed" — it is real kernel time. 1607 kernel launches/step.

| rank | launches/step | ms/step | µs/launch | share | reading |
|---|---|---|---|---|---|
| 1 | 272 | **28.12** | 103.4 | **66.6%** | main GEMM (matches the §2 microbench 28.5 ms) ⇒ only 5.3 ms above the 22.8 ms floor |
| 2 | 15 | 2.95 | 198.4 | 7.0% | large single kernel, to be named |
| 3 | 48 | 1.51 | 31.7 | 3.6% | 48/step family = the 48 GDN layers |
| 4 | 48 | 1.48 | 30.7 | 3.5% | same |
| 5 | 131 | 1.29 | 9.8 | 3.0% | small kernels, high frequency |
| 6 | 5 | 0.78 | 158.1 | 1.9% | |
| 7 | 1 | 0.60 | 599.8 | 1.4% | once/step, 600 µs single (step anchor) |
| 8 | 181 | 0.58 | 3.2 | 1.4% | |
| 9 | 127 | 0.54 | 4.2 | 1.3% | |
| 10 | 2 | 0.52 | 262.9 | 1.2% | |

**Corrected conclusions:**

1. **GDN is not the hotspot.** The 48/step family (16302 + 15696 + 16301 + 16303 + 1939 +
   253 + 1733) totals only **~4.0 ms/step**, not the 63 ms the eager-mode profiler shows.
   The eager 63 ms is mostly **Python/launch overhead** that does not exist in graph mode.
   The eager table is only valid for *relative* comparison within eager.
2. **The main GEMM is near its limit** (28.1 vs 22.8 floor) ⇒ to go faster you must reduce
   the weights read (quantization / sparsity), not optimize the kernel.
3. The real movable target is the **non-GEMM 14.1 ms/step tail**: the "big and odd" kernels
   (rank 2, 7, 9, 5) are the most worth naming and digging into.
4. **Kernel id → name is missing** in the `.dat` (rocprofv3 resolves names only at output
   time); you must let the scheduler subprocess exit cleanly to get names. Use the built-in
   profiler route instead (`tooling/phase3_prof.sh`).

---

## 3. The speed ladder (PP / TG / accept)

| Stage | PP (4k / 8k / 20k) | TG (code / prose / 20K) | accept (code / prose / 20K) |
|---|---|---|---|
| Phase 0 (stock AWQ) | 2057.5 / 1847.6 / 1378.3 | — | — |
| + atiled fused-norm (`SGLANG_MXFP4_ATILED_MIN_M=513`) | 2256.6 / 2017.5 / 1474.3 | — | — |
| **+ r4d GDN prefill (`SGLANG_R4D_GDN_MIN_T=513`)** | **2353.5 / 2130.2 / 1529.9** | **117.6 / 32.6 / 57.9** | **4.83 / 1.506 / 3.08** |

The earlier 100.7 figure (code, `hist.py` ruler) and the 117.6 (code, `bench.py` ruler) are
**not directly comparable** — different rulers. The `bench.py` ruler is the one used from
2026-10-07 onward.

**Content determines accept length, and it varies 3× across scenarios** — "code 117 /
prose 33" on the same engine is normal. Do not mistake a scenario difference for an engine
regression. The Paiton/vLLM reference 88.9 t/s is a *mixed-scenario* average; do not compare
it against prose.

---

## 4. Verify-M independence

The verify step's **M** (how many tokens are verified at once) barely affects the time.
`libr4d` MXFP4 GEMM microbench over the real TP1 shapes (`BENCH_MS=1`, 64 layers summed
per step):

| M | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| GEMM ms/step | 27.8 | 27.7 | 28.1 | **28.5** | 29.3 |

**M=8→4 saves only 0.4 ms.** Consequences (all confirmed by measurement):

- `--speculative-dflash-block-size 4/6` is a **net loss** — it saves no time and cuts the
  accept length.
- DSpark's "variable/compact verify" has **≈0 value at `MAX_RUN=1`** (its win is at batch B,
  where B×K tokens are verified together).
- `SGLANG_SIMULATE_ACC_LEN` **cannot** be used to calibrate M: it rewrites `accept_len` /
  `commit_lens` *after* verify, leaving `block_size = candidates.shape[1]` unchanged.

---

## 5. Three-way per-prompt alignment (code00, same prompt)

| Engine | accept | ms/step | tokens/s |
|---|---|---|---|
| SGLang MXFP4 (this repo) | **4.83** | **44.7** | **117.5** |
| SGLang 3-bit (W3A4) | 2.753 | 46.3 | 99.0 |
| Paiton/vLLM 3-bit | 2.876 | **29.5** | 97.5 |

**3-bit makes *our* step time go *up* (44.7 → 46.3 ms) while accept collapses
(4.83 → 2.75) — a loss on both axes.** Paiton with the same 3-bit also collapses accept to
2.88, but its step is 29.5 ms ⇒ **the 16.8 ms/step gap is entirely in the non-GEMM segment,
independent of the weight format.** The real target is that 15–17 ms/step non-GEMM tail.

---

## 6. The 3-bit (W3A4) line — why it was rejected

The 3-bit weight format has a real **byte-count** advantage (3k/8 vs 4k/8 bytes per weight):
at M=8, `mlp_down` 17408×5120 → w3 gemv **62.3 / 70.2 µs** vs radiance decode 87.8 /
mxfp4a8 92.6 (**25–34% faster**); `in_proj_qkv` 5120×10240 → **40.9 µs** vs 60.1 / 73.2
(**47–79% faster**). The effective bandwidth 476–537 GB/s ≈ radiance's 507–521 GB/s ⇒
**the whole win is the byte count, not the code.** Extrapolated: main GEMM 28.1 ms
(14.40 GB @ 512 GB/s) → 11.83 GB @ 537 GB/s ≈ **22.0 ms ⇒ −6.1 ms/step ⇒ step 41.2 → 35.1 ms
⇒ code TG 117.6 → ~138 (+17%)**.

But in the end-to-end test (3 rounds), **the microbench win is eaten by in-situ overhead and,
decisively, by the accept-rate collapse**: `accept_probe` code **2.753** (baseline 4.83),
prose 1.552 (1.506), 20K 2.695/2.37 (3.08/3.12). `SGLANG_MXFP4_FUSED_NORM=0` (removing the
"quantized norm → dequant → rotate → requant" double quantization) leaves accept **bit-for-bit
unchanged** (2.753) while TG drops to 92.9 ⇒ **the double quantization is *not* the cause;
FUSED_NORM=0 is rejected, the fused norm is kept.**

**Verdict: the 3-bit line is dead in SGLang.** The microbench byte-count win is real but
cannot survive the in-situ overhead + the accept-rate collapse. The real target is the
15–17 ms/step non-GEMM tail.

---

## 7. What still works (kept in production)

- **Our HIP verify-attention** (codename **r4d**, `libr4d` + `srt/layers/attention/r4d_verify.py`,
  `SGLANG_R4D_ATTN=1` on by default) — better accuracy than Triton (rel 0.0029 vs 0.048 vs
  float64 truth), short-ctx step 51.5 → 45.7 ms, 20K verify 77.9 → 39.3 ms.
- `SGLANG_VERIFY_BLOCK_N=64` (default 32 is tuned for 132 CU; only affects the Triton
  fallback path).
- `--sleep-on-idle`, `--image-processor-backend pil`, `--enable-cache-report`, `slru` eviction.
- 2026-10-07 additions: `SGLANG_MXFP4_ATILED_MIN_M=513` (layernorm fusion) and
  `SGLANG_R4D_GDN_MIN_T=513` (GDN prefill trio `gdn_conv_prep` + `gdn_kkt_solve` +
  `gdn_chunk_scan`, `srt/layers/attention/linear/r4d_gdn.py`).

**Two silent pitfalls when wiring the libr4d GDN kernels** (they do not crash, but the output
is entirely wrong):

1. The libr4d kernel does `if (slot <= 0) return;` (vLLM's NULL_BLOCK_ID convention). SGLang's
   intermediate scratch row index **starts at 0** (at `MAX_RUN=1`, `intermediate_ssm` has only
   2 rows, and request 0 uses row 0) → the kernel skips the whole thing, output stays
   uninitialized garbage, snapshots all 0, commit scatters 0 back to state: TG collapses to
   21–22, accept 1.4. **Fix: offset the slot index by +1, back the `state`/`cstate` pointers
   by one slot/row** (the actual accessed address is unchanged, never trips the null check).
2. The candidate snapshot index table must carry `+t`:
   `(srow*draft).repeat_interleave(draft)` writes all `draft` candidates into the same slot
   (parity signature: only u0 nonzero, and exactly equal to the reference's last candidate).
   Correct form: `srow[:,None]*draft + arange(draft)`.
