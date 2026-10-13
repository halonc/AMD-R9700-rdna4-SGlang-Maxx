#!/bin/bash
# Produce the two build inputs the Dockerfile needs:
#   sgl-v0520-patched/  = community RDNA4 base + our 5 patches
#   libr4d-patched/     = third-party libr4d + our 20-line patch
#
# Usage:
#   ./prepare-tree.sh /path/to/sgl-v0520-community-base /path/to/libr4d-checkout
#
# Prerequisites (see ../docs/DEPENDENCIES.md):
#   * the community RDNA4 base must already contain the 72-patch set (10-01
#     version, commit 784216d5c5 of mattbucci's repo).
#   * the libr4d checkout must be the StillDeadcode/libr4d Codeberg repo.

set -e

BASE="${1:?usage: prepare-tree.sh <sglang-community-base> <libr4d-checkout>}"
LIBR4D="${2:?usage: prepare-tree.sh <sglang-community-base> <libr4d-checkout>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# --- SGLang: community base + our 5 new-file patches -------------------------
OUT_SGL="$(pwd)/sgl-v0520-patched"
rm -rf "$OUT_SGL"
cp -a "$BASE" "$OUT_SGL"
cd "$OUT_SGL"
git init -q
git add -A
git commit -q -m "community RDNA4 base"
for p in 001-r4d-mxfp4-core 002-r4d-w4 003-r4d-w3 004-r4d-gdn 005-r4d-verify; do
    git am -p1 < "$REPO/patches/sglang/$p.patch"
done

# Mixed files (modified by both the community set and us): copy our in-tree
# versions over. This is correct ONLY if your base is the same 10-01 community
# patch set — otherwise reconcile by hand (see ../patches/PATCHSET.md §4).
cp -a "$REPO/patches/sglang/reference/python/sglang/srt/." python/sglang/srt/
echo "OK  sgl-v0520-patched: $(git log --oneline | wc -l) commits + reference files"

# --- libr4d: third-party base + our 20-line patch -----------------------------
OUT_LIB="$(pwd)/libr4d-patched"
rm -rf "$OUT_LIB"
cp -a "$LIBR4D" "$OUT_LIB"
cd "$OUT_LIB"
git init -q
git add -A
git commit -q -m "libr4d base"
git am -p1 < "$REPO/patches/libr4d/001-sglang-kv-layout.patch"
echo "OK  libr4d-patched"

echo
echo "Both trees ready. Next: docker compose -f $HERE/docker-compose.yml build"
