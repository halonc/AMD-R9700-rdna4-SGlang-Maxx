# Measurement Methodology

How to measure so the numbers actually mean something. Most "the numbers don't match"
problems are discipline problems, not engine problems.

---

## 1. The ruler

From 2026-10-07 onward, the unified ruler is `tooling/bench.py`:

- **Same prompt set** across all runs (code / Chinese prose / 20K technical Chinese).
- **Streaming**, `temperature=0` (greedy), `chat_template_kwargs:{enable_thinking:false}`.
- `max_tokens=256`, `ignore_eos`.
- **3 passes, median** (accept length jitters ±9% run-to-run; a single run is not a number).
- Prints `TG` (tokens/s), `PP` (prefill tokens/s), and an **accept probe**
  (`accept_probe=`) on the same prompt.

The earlier `hist.py` ruler (the 100.7 figure) and `bench.py` (the 117.6 figure) are
**not directly comparable** — different rulers. Do not mix them.

**Never re-run the full GSM8K-300 set** for speed work — it is a quality gate, not a speed
gate. Quality was already verified: AWQ vs MXFP4 on the same problem set shows no
statistical difference.

---

## 2. Measurement discipline (the numbers don't match? it's usually here)

1. **Look at `#running-req` / `#queue-req` first.** `MAX_RUN=1` — any concurrent request
   distorts TG. The log line `Decode batch, ... #queue-req: 0` is the only usable state.
2. **The first long prefill after `--sleep-on-idle` wake-up is ~4× slower.** (Measured:
   82K cold prefill first request only 370 tok/s; after wake-up, same conditions
   6.4K=1,694 / 27K=1,648 / 66K=1,140 tok/s.) **Always run a warm-up request before
   measuring**, or you will mistake the wake-up penalty for an "engine regression" or
   "disk noise."
3. **Accept length jitters ±9% run-to-run** — at least 2 runs; the ruler does 3.
4. **Eager and graph numbers are not comparable**; neither are numbers from different
   launch scripts (with/without warm-up loop, with/without `--sleep-on-idle`).
5. **Microbench per-rep sync inflates small kernels by ~25 µs** (host launch dominates).
   Calibration: null kernel + sync = 4.3 µs; single GEMM per-rep sync = 40 µs;
   back-to-back ×32 amortized = 11.6–14.7 µs. **Use the back-to-back amortized figure.**
6. **Repeated reads of the same weights hit L2/Infinity Cache** — the microbench will
   report >100% roof (especially mid shapes). Only use the M-trend and the clearly
   not-kernel-bound large shapes for conclusions.
7. **Input cap 178,522 tokens** — beyond that, HTTP 400 (easiest to hit when building a
   long prompt to test long context).
8. **Full-prompt logprobs OOM-crash the service** (`logprob_processor.py:585`,
   "Tried to allocate 628.00 MiB") → use a chunked NLL probe (CH=340).
9. Log `\r` flooding: `tr "\r" "\n" < log | grep -a …`. About 60 libavutil/torchcodec/
   cutlass/amdsmi tracebacks are harmless; do not mistake them for a fault.

---

## 3. A/B operation discipline (change a parameter, restart)

- **Kill and start are two separate ssh calls.** Escape the pkill pattern to avoid
  killing yourself: `"sglang\.launch_serve[r]"`, `"capture_prox[y]\.py"`,
  `"bench_mxfp4_msweep\.py"`.
- After pkill, **sleep 25–30 s**, then confirm `mem_info_vram_used < 2 GB` before
  starting (normal release is ~60 MB).
- **The 8080 proxy has a watchdog** that re-launches the capture proxy every 20 s.
  Before testing 8081, **stop the watchdog first, then the proxy**
  (`pkill -f "run-captur[e].sh"` then `pkill -f "capture_prox[y]\.py"`), or real traffic
  will pollute the data. (Once two watchdogs were running at the same time — only start
  one when recovering.)
- Start: `cd ~ && setsid nohup ./launch-xxx.sh > /tmp/sgl-xxx.log 2>&1 < /dev/null &`
  (background job). **Cold start 120–220 s** (longer with a big draft).
- The trial script = a copy of the production launch script with only the delta changed,
  logged to its own file. After the test, **roll back production and re-verify
  end-to-end** (run code + prose on the production port, compare to the baseline).
- **Production red lines** (check before any change): `CHUNK=2048`, `MEMF≤0.94`,
  `MAX_RUN=1`, `MAMBA=16`, `CTX=182000`, `--enable-cache-report`, `--sleep-on-idle`,
  `--image-processor-backend pil`, `RADIANCE_MXFP4_WPERM=1`, weights on the NAS, DPM `auto`.
- **Another session may be working on the same machine** (vision path, MAMBA A/B,
  `--sleep-on-idle`): before concluding, check the process start time and script mtime,
  and do not mistake someone else's change for your own regression.

---

## 4. The profiler route

**Preferred = the built-in SGLang profiler** (the route that actually produced data on
2026-10-07 Phase 3):

```bash
curl -s -X POST http://127.0.0.1:8081/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/sglprof","activities":["GPU"],"with_stack":false,"record_shapes":false,"profile_prefix":"p3"}'
#   …run a workload…
curl -s -X POST http://127.0.0.1:8081/stop_profile
#   output /tmp/sglprof/p3-<ts>-TP-0.trace.json.gz (chrome trace; aggregate by name
#   to get per-step kernel time)
```

Aggregate with `tooling/trace_agg.py`; per-step attribution in `FINDINGS.md` §2.
**The profiler has real overhead** (same request 2.808 s vs 2.369 s) ⇒ use it only for
attribution, never for speed measurement.

**The rocprofv3 route** (2026-10-06, kept as an environment fact): the tool is
`rocprofv3` (ROCm 7.x = rocprofiler-sdk 1.1.0), not v2; by default it does not trace
kernels, you must set `ROCPROF_KERNEL_TRACE=1`. All four capture methods failed in
practice; the only workable one is **offline parsing of the `.dat`** (pure CPU, no GPU,
production unaffected). **Next time, use the built-in profiler and do not touch rocprofv3.**

**Discipline:** during profiling, the production service on the same card must be stopped;
**confirm `mem_info_vram_used < 2 GB` before starting** (2026-10-06 we raced at 34 GB,
an old process still releasing, and hit `torch.OutOfMemoryError`). `pkill -f "rocprofv3"`
kills your own ssh session (the command line contains that word) ⇒ use `pgrep -f
"rocpro[f]v3"` or kill by PID. `LD_PRELOAD` pollutes every command in the same shell
(`ls`/`rm` get traced too) ⇒ put it in a script, never `export` it in an interactive shell.

---

## 5. The probes and tools

| Purpose | Path |
|---|---|
| Accept-length histogram + TG (code/prose/20K, 2× each) | `hist.py` (not in this repo; the ruler is `tooling/bench.py`) |
| Fixed 6-example greedy output capture (quality regression) | `outcap.py` |
| Chunked NLL (safe, CH=340) | `nll_safe.py` |
| Multi-length stress (4K/20K/56K/131K) | `stress_len.py` |
| GEMM M-sweep (libr4d real shapes) | `libr4d/bench_mxfp4_msweep.py` (`BENCH_MS=1`; `torch.cuda.set_per_process_memory_fraction(0.008)` to avoid stealing production VRAM) |
| Layer time / phase time | env `SGLANG_LAYER_PROF=1` (eager only), `SGLANG_DFLASH_PHASE_PROF=1` |
| **Unified 3-pass ruler (from 2026-10-07)** | `tooling/bench.py` (`--base http://127.0.0.1:8081 --engine sglang --passes 3 --out results-<tag>.jsonl`; writes `.summary.json` and prints `SUMMARY` + `accept_probe=`) |
| GDN prefill/decode/verify unit parity and microbench | `gdn_parity.py`, `gdn_decode_parity.py`, `gdn_decode_bench.py`, `gdn_verify_parity.py` |
| Baseline table / rollback script | `baseline-20261007.md`, `rollback-phase1-atiled.sh` |

**Run probes/microbench with the service's interpreter**: `miniforge3/envs/sglang-rdna4-single/bin/python`
(system `python3` has no torch). Find it with `readlink -f /proc/<pid>/exe`.

---

## 6. Environment kill-switches

`SGLANG_R4D_ATTN` (default 1), `SGLANG_R4D_DBG/_DUMP/_ZEROSCRATCH`, `R4D_SO`,
`SGLANG_MXFP4_DRAFT`, `SGLANG_VERIFY_BLOCK_N/_MAX_SPLITS/_MIN_SPLITS/_SPLIT_TH`,
`SGLANG_GDN_WARPS/_BV/_STAGES`, `SGLANG_LAYER_PROF`, `SGLANG_DFLASH_PHASE_PROF`,
`SGLANG_SIMULATE_ACC_LEN/_METHOD/_TOKEN_MODE`, `SGLANG_RAGGED_VERIFY_MODE`
(`static`/`cap-accept`/`compact`, **DSpark only**).

2026-10-07 additions: `SGLANG_MXFP4_ATILED_MIN_M` (production **513**),
`SGLANG_R4D_GDN_MIN_T` (production **513**), `SGLANG_R4D_GDN_DECODE_MIN_T` /
`SGLANG_R4D_GDN_VERIFY_MIN_T` (**default 0, rejected**), `SGLANG_R4D_GDN_DEBUG`
(one-shot diagnostic print). The production launch script sets only the first two.

**Fastest whole rollback**: set `SGLANG_MXFP4_ATILED_MIN_M` and `SGLANG_R4D_GDN_MIN_T` to 0
to return to the Phase 0 baseline (the switch=0 baseline numbers were reproduced, causality proven).
