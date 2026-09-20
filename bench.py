"""Paired interleaved timer for MLX kernels. Method lifted from the chunkwise harness.

    from bench import paired, probe
    paired(candidate, baseline, bytes_moved=...)   # candidate/baseline are zero-arg fns returning an mx.array
    probe()                                        # measure this machine's copy bandwidth once, stored in machine.json

Why interleaved: thermal drift hits A and B equally, so the median of per-pair ratios is
trustworthy even on a laptop. CV > 8% means the number is noise: pin fans, close apps, rerun.
Arc 2 (weeks 6-10) uses the full harness at
~/myproj/apple_job_track/code/chunkwise/harness.py instead of this file.
"""
import json, os, statistics, time
import mlx.core as mx

MACHINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machine.json")


def _time(fn, iters):
    mx.eval(fn()); mx.eval(fn())                  # warm: JIT compile, then cache
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())                             # eval each: MLX is lazy, an un-evaled result costs nothing
    return (time.perf_counter() - t) / iters


def probe(n=1 << 26):
    """Copy bandwidth (GB/s) of this machine for an fp32 array of n elements."""
    a = mx.random.normal((n,)); mx.eval(a)
    s = _time(lambda: a + 0, 50)
    gbps = 2 * n * 4 / s / 1e9                    # read + write
    json.dump({"copy_gbps": gbps, "when": time.strftime("%Y-%m-%d")}, open(MACHINE, "w"))
    print(f"copy bandwidth {gbps:.0f} GB/s -> {MACHINE}")
    return gbps


def paired(candidate, baseline, bytes_moved=None, pairs=15, iters=20):
    """Median per-pair speedup of candidate over baseline. Prints CV and roofline %."""
    ratios, cand = [], []
    for _ in range(pairs):
        b = _time(baseline, iters); c = _time(candidate, iters)
        ratios.append(b / c); cand.append(c)
    med_c = statistics.median(cand)
    cv = statistics.pstdev(cand) / statistics.mean(cand)
    line = f"speedup x{statistics.median(ratios):.2f}  candidate {med_c*1e6:.1f} us  CV {cv*100:.1f}%"
    if bytes_moved and os.path.exists(MACHINE):
        peak = json.load(open(MACHINE))["copy_gbps"]
        line += f"  roofline {bytes_moved / med_c / 1e9 / peak * 100:.0f}% of {peak:.0f} GB/s"
    if cv > 0.08:
        line += "  ** VARIANCE WARNING: pin fans, close apps, rerun **"
    print(line)
    return statistics.median(ratios)


if __name__ == "__main__":
    # self-check: a kernel timed against itself must report ~1.0x
    a = mx.random.normal((1 << 24,)); mx.eval(a)   # 64 MB: dispatch overhead < 5%
    r = paired(lambda: a * 2, lambda: a * 2, bytes_moved=2 * a.nbytes)
    assert 0.8 < r < 1.25, r
    print("bench.py ok")
