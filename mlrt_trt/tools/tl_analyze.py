import re,sys
rows=[]
for l in open(sys.argv[1] if len(sys.argv)>1 else 'tl_run.log',errors='ignore'):
    m=re.match(r'TL inst=(\d+) n=(\d+) t0=([\d.]+) t1=([\d.]+) t2=([\d.]+) t3=([\d.]+) t4=([\d.]+) t5=([\d.]+)',l)
    if m: rows.append((int(m[1]),int(m[2]))+tuple(float(m[i]) for i in range(3,9)))
rows.sort(key=lambda r:r[2])
rows=rows[100:-20]   # steady state
n=len(rows); print("calls analysed:",n)
import statistics as st
names=["H2D(raw+ts)","pack","TensorRT","unpack","D2H"]
for k,nm in enumerate(names):
    v=[r[3+k]-r[2+k] for r in rows]
    print(f"  {nm:12s} mean {st.mean(v):6.3f} ms  (min {min(v):.3f}, max {max(v):.3f})")
def union(iv):
    iv=sorted(iv); tot=0; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    return tot+ce-cs
def intersect(a,b):
    # total length of intersection of two lists of intervals (each merged first)
    def merge(iv):
        iv=sorted(iv); out=[list(iv[0])]
        for s,e in iv[1:]:
            if s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
            else: out.append([s,e])
        return out
    A,B=merge(a),merge(b); i=j=0; tot=0
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]); hi=min(A[i][1],B[j][1])
        if hi>lo: tot+=hi-lo
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return tot
w0=min(r[2] for r in rows); w1=max(r[7] for r in rows); W=w1-w0
comp=[(r[3],r[6]) for r in rows]                          # pack..unpack end
copy=[(r[2],r[3]) for r in rows]+[(r[6],r[7]) for r in rows]
u_c=union(comp); u_x=union(copy); ov=intersect(comp,copy)
busy=union(comp+copy)
print(f"window {W:.1f} ms for {n} calls -> {W/n:.2f} ms per inference ({1000*n/W:.1f} inferences/s)")
print(f"  compute-phase busy (pack..unpack): {100*u_c/W:5.1f}% of window")
print(f"  copy-phase busy (H2D + D2H)      : {100*u_x/W:5.1f}% of window")
print(f"  copy overlapping compute         : {100*ov/W:5.1f}% of window")
print(f"  nothing running (idle gaps)      : {100*(W-busy)/W:5.1f}% of window")
# concurrency: how many calls in flight at once (t0..t5)
ev=[]
for r in rows: ev+= [(r[2],1),(r[7],-1)]
ev.sort(); cur=0; last=ev[0][0]; hist={}
for t,d in ev:
    hist[cur]=hist.get(cur,0)+t-last; cur+=d; last=t
print("  calls in flight (time share):",{k:f"{100*v/W:.0f}%" for k,v in sorted(hist.items())})
# per-instance back-to-back gap
byi={}
for r in rows: byi.setdefault(r[0],[]).append(r)
for i,rs in byi.items():
    rs.sort(key=lambda r:r[2]); g=[rs[k+1][2]-rs[k][7] for k in range(len(rs)-1)]
    print(f"  instance {i}: {len(rs)} calls, gap between its calls: mean {st.mean(g):.2f} ms, median {st.median(g):.2f}")
