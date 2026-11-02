import re, sys, statistics as st
rows = []
for l in open(sys.argv[1] if len(sys.argv) > 1 else 'tl_run.log', errors='ignore'):
    m = re.match(r'TL inst=(\d+) n=(\d+) .* h_acq=([\d.-]+) h_sync=([\d.-]+) h_in=([\d.-]+)', l)
    if m:
        rows.append((int(m[1]), int(m[2]), float(m[3]), float(m[4]), float(m[5])))
rows = rows[100:-20]
rows.sort(key=lambda r: r[2])
ev = []
for r in rows:
    ev += [(r[2], 1), (r[3], -1)]
ev.sort()
cur = 0
last = ev[0][0]
hist = {}
for t, d in ev:
    hist[cur] = hist.get(cur, 0) + t - last
    cur += d
    last = t
W = ev[-1][0] - ev[0][0]
print("host threads between ticket-acquire and GPU-sync:", {k: f"{100 * v / W:.0f}%" for k, v in sorted(hist.items())})
rs = sorted(rows, key=lambda r: r[2])
g = [rs[k + 1][2] - rs[k][3] for k in range(len(rs) - 1)]
print(f"gap from a call's GPU-sync to the next call's ticket-acquire: mean {st.mean(g):.2f} ms, median {st.median(g):.2f}, min {min(g):.2f}, max {max(g):.2f}")
v = [r[2] - r[4] for r in rows]
print(f"host time from getframe start to ticket-acquire (children + semaphore): mean {st.mean(v):.2f} ms")
# arrival rate of calls at the plugin: getframe start times
a = sorted(r[4] for r in rows)
d = [a[i + 1] - a[i] for i in range(len(a) - 1)]
print(f"calls arrive at the plugin every {st.mean(d):.2f} ms on average (median {st.median(d):.2f}); GPU needs ~20.1 ms each")
