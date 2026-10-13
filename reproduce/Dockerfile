# SGLang v0.5.20 (RDNA4 fork) + our delta + libr4d, on ROCm 7.2.
#
# Build inputs (placed in this directory before `docker compose build`):
#   sgl-v0520-patched/   the SGLang v0.5.20 tree WITH the 72-patch community
#                        RDNA4 set AND our 5 patches already applied.
#                        Produce it with ./prepare-tree.sh (see README.md).
#   libr4d-patched/      the StillDeadcode/libr4d checkout WITH our 20-line
#                        patch already applied (prepare-tree.sh does it too).
#
# The third-party radiance_mxfp4_fp8.so is NOT in the image — it is mounted in
# at runtime (see docker-compose.yml). Without it the headline TG is ~24-32%
# lower (Open config path).

ARG BASE_IMAGE=rocm/pytorch:rocm7.2
FROM ${BASE_IMAGE}

WORKDIR /opt

# --- SGLang fork (community base + our delta) --------------------------------
COPY sgl-v0520-patched /opt/sglang
RUN pip install -e /opt/sglang/python

# --- libr4d (third-party + our 20-line patch) ---------------------------------
COPY libr4d-patched /opt/libr4d
RUN cd /opt/libr4d && ./build.sh   # produces r4d.so

# --- environment ---------------------------------------------------------------
ENV PATH=/opt/rocm/bin:$PATH \
    ROCM_PATH=/opt/rocm \
    PYTHONPATH=/opt/libr4d \
    SGLANG_DIR=/opt/sglang
