#!/bin/bash
# Capture a decode-stage kernel profile with SGLang's built-in torch.profiler
# (same production config, same code00 prompt). The profiler has real overhead
# (~18% on this workload) — use it for attribution only, never for speed.
#
# Usage:
#   BASE=http://127.0.0.1:8081 PROMPT_FILE=code00.jsonl ./phase3_prof.sh
#
# PROMPT_FILE is a .jsonl where each line is {"text": ...}; the first line is used.
set -e

BASE="${BASE:-http://127.0.0.1:8081}"
PROMPT_FILE="${PROMPT_FILE:-code00.jsonl}"
PY="${PY:-python}"

rm -rf /tmp/sglprof; mkdir -p /tmp/sglprof

P=$("$PY" -c "
import json
r=[json.loads(l) for l in open('$PROMPT_FILE')][0]
print(json.dumps(r['text'], ensure_ascii=False))
")

curl -s -m 20 -X POST "$BASE/start_profile" -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/sglprof","activities":["GPU"],"with_stack":false,"record_shapes":false,"profile_prefix":"p3"}'
echo

T=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen3.8-27B-MXFP4-DFlash2\",\"messages\":[{\"role\":\"user\",\"content\":$P}],\"max_tokens\":256,\"temperature\":0,\"ignore_eos\":true,\"stream\":false,\"return_meta_info\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  -o /tmp/sglprof/resp.json -w '%{time_total}')
echo "request t=$T"

"$PY" -c "
import json; d=json.load(open('/tmp/sglprof/resp.json'))
mi=(d.get('choices') or [{}])[0].get('meta_info') or {}
print('meta:', {k:v for k,v in mi.items() if 'spec' in k or 'completion' in k or 'prompt' in k})"

curl -s -m 60 -X POST "$BASE/stop_profile"
echo
sleep 6; ls -la /tmp/sglprof | tail -8
