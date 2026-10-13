import gzip, json, glob, collections, sys, re
f = sorted(glob.glob("/tmp/sglprof/*.trace.json.gz"))[-1]
d = json.load(gzip.open(f, "rt"))
ev = [e for e in (d["traceEvents"] if isinstance(d, dict) else d) if e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 53
agg = collections.defaultdict(lambda: [0.0, 0])
for e in ev:
    n = e["name"]
    n = re.sub(r"<[^<>]{0,80}>", "<>", n)
    n = n.replace("void ", "").replace("at::native::", "").replace("(anonymous namespace)::", "")
    a = agg[n]; a[0] += e.get("dur", 0); a[1] += 1
tot = sum(a[0] for a in agg.values())
print("kernel 总时长 %.1f ms / %d 步 = **%.2f ms/步**   不同内核 %d 种" % (tot/1e3, STEPS, tot/1e3/STEPS, len(agg)))
print("%-62s %9s %7s %9s %7s" % ("内核", "总ms", "次数", "ms/步", "次/步"))
for n, (us, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:26]:
    print("%-62s %9.2f %7d %9.3f %7.2f" % (n[:62], us/1e3, c, us/1e3/STEPS, c/STEPS))
print()
print("--- 次数正好是 %d 的倍数（纯 decode 内核）---" % STEPS)
sub = [(n, us, c) for n, (us, c) in agg.items() if c % STEPS == 0]
sub.sort(key=lambda x: -x[1])
for n, us, c in sub[:18]:
    print("%-62s %9.2f %7d %9.3f" % (n[:62], us/1e3, c, us/1e3/STEPS))
