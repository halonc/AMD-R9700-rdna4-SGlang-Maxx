# -*- coding: utf-8 -*-
"""统一口径 bench：同一 prompt 集打 SGLang(8081) 或 Paiton(8080)。
流式计时：TTFT=首内容块；decode TG=completion_tokens/(t_end-t_first)；PP=prompt_tokens/TTFT。
greedy(temperature=0)、思考关、max_tokens=256、ignore_eos（不支持则自动降级并标注）。
SGLang 额外跑 hist 式 accept 探针；Paiton 读 /metrics 投机计数器差分。"""
import json, time, urllib.request, statistics, argparse, os, sys, re

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    rows = []
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def get_model(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        d = json.load(r)
    return d["data"][0]["id"]

def post_stream(base, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; usage = None; nchunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                content = delta.get("content")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    nchunks += 1
    t_end = time.perf_counter()
    return t0, t_first, t_end, usage, nchunks

def post_once(base, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.perf_counter() - t0

def probe_ignore_eos(base, model):
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4, "temperature": 0, "ignore_eos": True, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        post_once(base, body, 60); return True
    except Exception:
        return False

def metrics_snapshot(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out = {}
    for m in re.finditer(r"^([a-zA-Z_:]+(?:_total|_count)?)\s+([0-9.eE+]+)\s*$", txt, re.M):
        name, val = m.group(1), float(m.group(2))
        if "spec" in name or "draft" in name or "accepted" in name:
            out[name] = val
    return out

def run_case(base, model, row, max_tokens, ignore_eos, engine):
    body = {"model": model, "messages": [{"role": "user", "content": row["text"]}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    t0, t_first, t_end, usage, nchunks = post_stream(base, body)
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or nchunks
    ttft = (t_first - t0) if t_first else None
    dec = (t_end - t_first) if t_first else None
    return {"id": row["id"], "scenario": row.get("scenario"), "depth": row.get("depth"),
            "prompt_tokens": pt, "completion_tokens": ct,
            "ttft_s": round(ttft, 3) if ttft else None,
            "decode_s": round(dec, 3) if dec else None,
            "tg": round(ct / dec, 1) if dec and ct else None,
            "pp": round(pt / ttft, 1) if ttft and pt else None}

def accept_probe(base, model):
    """SGLang hist 式 accept 探针（非流式 + return_meta_info），每场景首题 ×2。"""
    res = {}
    for scen, name in (("code", "prompts_code.jsonl"), ("prose", "prompts_prose.jsonl"),
                       ("20k", "prompts_20k.jsonl")):
        row = load(name)[0]
        body = {"model": model, "messages": [{"role": "user", "content": row["text"]}],
                "max_tokens": 256, "temperature": 0, "ignore_eos": True, "stream": False,
                "return_meta_info": True, "chat_template_kwargs": {"enable_thinking": False}}
        vals = []
        for _ in range(2):
            d, el = post_once(base, body)
            mi = (d.get("choices") or [{}])[0].get("meta_info") or {}
            if mi.get("spec_accept_length") is not None:
                vals.append(mi["spec_accept_length"])
        if vals:
            res[scen] = [round(v, 3) for v in vals]
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--engine", required=True, choices=("sglang", "paiton"))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = get_model(args.base)
    ignore_eos = probe_ignore_eos(args.base, model)
    print("engine=%s model=%s ignore_eos=%s" % (args.engine, model, ignore_eos), flush=True)
    decode_rows = load("prompts_code.jsonl") + load("prompts_prose.jsonl") + load("prompts_20k.jsonl")
    prefill_rows = load("prompts_prefill.jsonl")
    out = open(args.out, "w", encoding="utf-8")
    per_pass = {}
    for p in range(1, args.passes + 1):
        m0 = metrics_snapshot(args.base)
        rows = []
        # 热身（--sleep-on-idle 唤醒惩罚 / 引擎预热）：一次短请求
        run_case(args.base, model, {"id": "warm", "scenario": "warm", "text": "warmup"}, 32, ignore_eos, args.engine)
        for row in decode_rows:
            r = run_case(args.base, model, row, 256, ignore_eos, args.engine)
            r["engine"] = args.engine; r["pass"] = p
            rows.append(r); out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
            print("p%d %s %s: pt=%s ct=%s ttft=%s tg=%s" %
                  (p, r["scenario"], r["id"], r["prompt_tokens"], r["completion_tokens"],
                   r["ttft_s"], r["tg"]), flush=True)
        for row in prefill_rows:
            r = run_case(args.base, model, row, 1, ignore_eos, args.engine)
            r["engine"] = args.engine; r["pass"] = p
            rows.append(r); out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
            print("p%d prefill %s: pt=%s ttft=%s pp=%s" %
                  (p, r["id"], r["prompt_tokens"], r["ttft_s"], r["pp"]), flush=True)
        m1 = metrics_snapshot(args.base)
        delta = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1 if k not in m0 or True}
        tg_by = {}
        for r in rows:
            tg_by.setdefault(r["scenario"], []).append(r["tg"])
        med = {k: round(statistics.median([v for v in vs if v]), 1) for k, vs in tg_by.items() if any(v for v in vs)}
        pp_by = {}
        for r in rows:
            if r["scenario"] == "prefill":
                pp_by.setdefault(r["depth"], []).append(r["pp"])
        pp_med = {k: round(statistics.median([v for v in vs if v]), 1) for k, vs in pp_by.items() if any(v for v in vs)}
        per_pass[p] = {"tg": med, "pp": pp_med, "spec_metrics_delta": delta}
        print("PASS %d tg=%s pp=%s spec_delta=%s" % (p, med, pp_med, delta), flush=True)
    if args.engine == "sglang":
        acc = accept_probe(args.base, model)
        print("accept_probe=%s" % acc, flush=True)
    else:
        acc = None
    summary = {"engine": args.engine, "model": model, "ignore_eos": ignore_eos,
               "passes": per_pass, "accept_probe": acc,
               "tg_median_of_pass_medians": {
                   s: round(statistics.median([per_pass[p]["tg"][s] for p in per_pass if s in per_pass[p]["tg"]]), 1)
                   for s in ("code", "prose", "20k")},
               "pp_median_of_pass_medians": {
                   d: round(statistics.median([per_pass[p]["pp"][d] for p in per_pass if d in per_pass[p]["pp"]]), 1)
                   for d in ("4k", "8k", "20k")}}
    sp = args.out.replace(".jsonl", ".summary.json")
    json.dump(summary, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summary["tg_median_of_pass_medians"], ensure_ascii=False) + " " +
          json.dumps(summary["pp_median_of_pass_medians"], ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
