# Reproduction Guide

Step-by-step: from a clean machine to a running SGLang server with the finalized
configuration. Read [`../docs/DEPENDENCIES.md`](../docs/DEPENDENCIES.md) first — it
lists what you must obtain from third parties before you can start.

---

## Step 0 — Obtain the non-redistributable pieces

| Item | Where | Notes |
|---|---|---|
| SGLang v0.5.20 + 72-patch community RDNA4 set (10-01, commit `784216d5c5`) | from the author of `mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference` | **no license file — do not redistribute** |
| `libr4d` kernel library | `StillDeadcode/libr4d` on Codeberg | third-party; we apply only our 20-line patch |
| `radiance_mxfp4_fp8.so` (decode GEMM) | inside **paiton's public Docker image** | **no source anywhere**; without it, TG is ~24–32% lower |
| Qwen3.8-27B AWQ checkpoint | the checkpoint source | not redistributable |
| DFlash2 draft (`z-lab/Qwen3.8-27B-DFlash2`) | Hugging Face | 5-layer draft, chain topology |

---

## Step 1 — Convert the AWQ checkpoint to MXFP4

```bash
python convert-awq-to-mxfp4.py \
    --in  /data/models_sglang/qwen38-awq \
    --out /data/models_sglang/qwen38-mxfp4
```

The output has `weight_packed` (uint8 E2M1, `[N, K/2]`, fragment order) and
`weight_scale` (uint8 E8M0, `[K/32, N]`) for every AWQ Linear layer; everything else
is copied through unchanged. This is the same conversion the runtime does at load —
doing it offline means the runtime reads MXFP4 directly and never keeps the AWQ copy
resident.

---

## Step 2 — Build the two trees

```bash
./prepare-tree.sh /path/to/sgl-v0520-community-base /path/to/libr4d-checkout
```

This produces, in the current directory:

- `sgl-v0520-patched/` — the community base + our 5 new-file patches + the reference
  mixed files.
- `libr4d-patched/` — the third-party `libr4d` + our 20-line patch.

---

## Step 3 — Mount the radiance .so

```bash
# from paiton's public image:
docker pull <paiton-public-image>
docker create --name radiance-src <paiton-public-image>
mkdir radiance
docker cp radiance-src:/path/to/radiance_mxfp4_fp8.so radiance/
```

`docker-compose.yml` mounts it at `/opt/radiance/radiance_mxfp4_fp8.so`. **Skip this
step only if you are deliberately running the Open config path** (no radiance; ~24–32%
lower TG).

---

## Step 4 — Build and start

```bash
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d
```

Cold start is ~120–220 s (longer with a big draft). Verify:

```bash
curl -s http://127.0.0.1:8081/v1/models | grep MXFP4-DFlash2
```

---

## Step 5 — Measure with the unified ruler

```bash
python ../tooling/bench.py \
    --base http://127.0.0.1:8081 \
    --engine sglang \
    --passes 3 \
    --out results-<tag>.jsonl
```

The ruler prints `TG`, `PP`, and an `accept_probe=` line. **Always run a warm-up
request first** (the first long prefill after `--sleep-on-idle` wake-up is ~4× slower).
See [`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for the full measurement discipline.

---

## The expected numbers (with `radiance_mxfp4_fp8.so`)

| Metric | 4k | 8k | 20k |
|---|---|---|---|
| PP (prefill tokens/s) | 2353.5 | 2130.2 | 1529.9 |
| TG (code / prose / 20K) | 117.6 / 32.6 / 57.9 | — | — |
| accept (code / prose / 20K) | 4.83 / 1.506 / 3.08 | — | — |

Without the radiance binary (Open config path), expect ~24–32% lower TG.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| server dies at startup, "zero GPU" | `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` set to an **empty string** (an empty list is read as zero GPUs) | set them to a real card index (never `VAR=`) |
| garbage output, accept 1.0 | the GDN slot-index +1 offset or the `+t` snapshot index fix is missing | see `../docs/FINDINGS.md` §7 |
| OOM at 20K prefill with `CHUNK=8192` | SGLang's `mem_fraction_static` locks KV+mamba before counting peak transient memory (vLLM does not) | keep `CHUNK=2048` |
| TG looks like a regression after wake-up | the first long prefill after `--sleep-on-idle` is ~4× slower | run a warm-up request before measuring |
