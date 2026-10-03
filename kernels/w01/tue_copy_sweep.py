"""Week 1 Tue: copy kernel, sweep threadgroup width x elements per thread.

    .venv/bin/python kernels/w01/tue_copy_sweep.py

Prints GB/s per cell (read + write bytes / median time). Baseline row: MLX's own `a + 0`.
"""
import statistics, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import mlx.core as mx

N = 1 << 26                      # fp32 elements = 256 MB; a copy moves 512 MB
BYTES = 2 * N * 4
WIDTHS = [32, 64, 128, 256, 512, 1024]
ELEMS = [1, 2, 4, 8]

# One thread copies E consecutive elements. E is a compile-time template int,
# so the loop has a constant trip count the compiler can unroll.
kernel = mx.fast.metal_kernel(
    name="copy_sweep",
    input_names=["inp"],
    output_names=["out"],
    source="""
        uint base = thread_position_in_grid.x * E;
        for (uint i = 0; i < E; ++i) { out[base + i] = inp[base + i]; }
    """,
)

def copy(a, width, e):
    return kernel(inputs=[a], template=[("E", e)], grid=(N // e, 1, 1),
                  threadgroup=(width, 1, 1), output_shapes=[a.shape], output_dtypes=[a.dtype])[0]

def gbps(fn, trials=7, iters=5):
    mx.eval(fn()); mx.eval(fn())                       # JIT, then warm
    ts = []
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())                              # MLX is lazy: eval every iteration
        ts.append((time.perf_counter() - t) / iters)
    med = statistics.median(ts)
    return BYTES / med / 1e9, statistics.pstdev(ts) / statistics.mean(ts)

if __name__ == "__main__":
    a = mx.random.normal((N,)); mx.eval(a)
    assert mx.array_equal(copy(a, 256, 4), a).item(), "copy kernel is wrong"   # correctness before speed
    base, bcv = gbps(lambda: a + 0)
    print(f"baseline  a + 0 : {base:6.1f} GB/s  (CV {bcv*100:.1f}%)\n")
    print("GB/s      " + "".join(f"E={e:<7}" for e in ELEMS))
    worst_cv = 0.0
    for w in WIDTHS:
        row = []
        for e in ELEMS:
            g, cv = gbps(lambda: copy(a, w, e)); worst_cv = max(worst_cv, cv)
            row.append(f"{g:6.1f}{'*' if cv > 0.08 else ' '}  ")
        print(f"W={w:<6}  " + "".join(row))
    base2, _ = gbps(lambda: a + 0)
    print(f"\nbaseline again  : {base2:6.1f} GB/s   worst cell CV {worst_cv*100:.1f}%   (* = CV > 8%, treat as noise)")
