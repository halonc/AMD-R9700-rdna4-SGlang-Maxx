#!/bin/bash
# Single-card (GPU 0) launch for the Qwen3.8 RDNA4 stack — SGLang v0.5.20 + RDNA4 patches.
# Wrapped by launch-qwen38-nas.sh. All tunables come from the environment.
set -e

REPO=${REPO:-/opt/sglang}
MODEL="${MODEL:-/data/models_sglang/qwen38-mxfp4}"
PORT="${PORT:-8081}"
MEMF="${MEMF:-0.94}"
CG_ARGS="${CG_ARGS:---cuda-graph-max-bs-decode 4}"
CHUNK="${CHUNK:-2048}"
SPEC_ARGS="${SPEC_ARGS:-}"
MAMBA="${MAMBA:-16}"
CTX="${CTX:-182000}"
DECODE_STEPS="${DECODE_STEPS:-8}"
RADIX_ARGS="${RADIX_ARGS:-}"
ATTN="${ATTN:-triton}"
MAMBA_DTYPE="${MAMBA_DTYPE:---mamba-ssm-dtype float32}"
QUANT="${QUANT:-r4d_mxfp4}"
MAX_RUN="${MAX_RUN:-1}"
KVDTYPE="${KVDTYPE:-fp8_e4m3}"
# Text-only serving: drop the vision tower reservation.
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-0}"
# RDNA4 TP=1: the JIT store_cache path is broken; disable it at TP=1.
export SGLANG_RDNA4_DISABLE_STORE_CACHE="${SGLANG_RDNA4_DISABLE_STORE_CACHE:-1}"
# libr4d kernels (r4d.so) for the 4-bit WMMA path
export PYTHONPATH="/opt/libr4d:${PYTHONPATH:-}"
export SGLANG_DIR=${REPO}
export ROCM_PATH=/opt/rocm

# Single card: GPU 0 only.
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

export TVM_FFI_GPU_BACKEND=cuda
export SGLANG_ENABLE_SPEC_V2=0

echo "=============================================="
echo "Qwen3.8 RDNA4 single-card launch"
echo "Model: $MODEL  Port: $PORT  mem-fraction: $MEMF"
echo "=============================================="

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --kv-cache-dtype $KVDTYPE \
    --context-length $CTX \
    --mem-fraction-static "$MEMF" \
    --max-running-requests $MAX_RUN \
    --chunked-prefill-size $CHUNK \
    --num-continuous-decode-steps $DECODE_STEPS \
    --attention-backend $ATTN \
    --pre-warm-nccl \
    --watchdog-timeout 300 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --quantization $QUANT \
    --max-mamba-cache-size $MAMBA \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    $CG_ARGS \
    $SPEC_ARGS \
    $LM_ARGS \
    $RADIX_ARGS \
    $MAMBA_DTYPE
