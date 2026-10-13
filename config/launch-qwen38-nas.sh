#!/bin/bash
# Production launch wrapper: reads the model from a host /data mount
# (in the original deployment the model lives on a NAS, accessed through a
# local /data symlink to keep the path free of spaces).
#
# This is the finalized configuration that produced the headline numbers
# (see ../docs/FINDINGS.md). It wraps the single-card launch script.
set -e

export PORT=8081
export RADIX_ARGS="--enable-cache-report --radix-eviction-policy slru"
export MODEL=/data/models_sglang/qwen38-awq
export QUANT=awq MEMF=0.94 CTX=182000 MAMBA=16 CHUNK=2048 MAX_RUN=1
export LM_ARGS='--mamba-full-memory-ratio 0.12 --mamba-radix-cache-strategy extra_buffer --page-size 16 --served-model-name Qwen3.8-27B-MXFP4-DFlash2 --tool-call-parser qwen3_coder --image-processor-backend pil --sleep-on-idle'
export CG_ARGS=--cuda-graph-max-bs-decode=4
export SPEC_ARGS='--speculative-algorithm DFLASH --speculative-draft-model-path /data/models_sglang/qwen38-dflash2'

# --- Tuning kill-switches (see ../reproduce/launch.env for the full table) ---
export RADIANCE_MXFP4_WPERM=1 SGLANG_EMBED_FP8=1
export SGLANG_RDNA4_DISABLE_STORE_CACHE=1
export SGLANG_R4D_ATTN=1
# draft fp8->MXFP4: OFF (accept 0.68 -> 0.25, net loss — see ../docs/DEAD-ENDS.md row 5)
export SGLANG_MXFP4_DRAFT=0
export SGLANG_VERIFY_BLOCK_N=64
# Phase 1 (2026-10-07): atiled prefill GEMM for M>=513. PP 4K +9.7% / 8K +9.2% / 20K +7.0%;
# decode and accept bit-identical. Set 0 to fully revert (kill-switch).
export SGLANG_MXFP4_ATILED_MIN_M=513
# Phase 2 (2026-10-07): libr4d fused GDN prefill (conv_prep+kkt_solve+chunk_scan),
# only for prefill chunks >= 513 tokens. PP 4K 2256.6->2353.5 / 8K 2017.5->2130.2 /
# 20K 1474.3->1529.9. Set 0 to revert to FLA Triton.
export SGLANG_R4D_GDN_MIN_T=513

exec ./launch-qwen38-single.sh
