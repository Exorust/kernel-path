"""Week 1 Wed: half-precision GEMV, y = W @ x, one simdgroup per row.

    .venv/bin/python kernels/w01/wed_gemv.py

Layout A: lane l reads columns l, l+32, l+64, ...  (32 lanes read 32 neighbours per step)
Layout B: lane l reads columns 128*l .. 128*l+127  (32 lanes read values 128 apart per step)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import mlx.core as mx
import bench

N = K = 4096
BYTES = N * K * 2                                   # every weight read once, 2 bytes each

def make(name, loop):
    return mx.fast.metal_kernel(
        name=name, input_names=["W", "x"], output_names=["out"],
        source=f"""
            uint lane = thread_position_in_grid.x;      // 0..31: threadgroup is exactly one simdgroup
            uint row  = thread_position_in_grid.y;
            const device half* w = W + row * KK;
            float acc = 0.0f;                           // fp32 accumulation, lives in a register
            {loop}
            float total = simd_sum(acc);                // add the 32 partial sums
            if (lane == 0) out[row] = total;
        """)

kA = make("gemv_a", "for (uint c = lane; c < KK; c += 32) acc += float(w[c]) * float(x[c]);")
kB = make("gemv_b", "uint lo = lane * (KK / 32); for (uint c = lo; c < lo + KK / 32; ++c) acc += float(w[c]) * float(x[c]);")

def gemv(k, W, x):
    return k(inputs=[W, x], template=[("KK", K)], grid=(32, N, 1), threadgroup=(32, 1, 1),
             output_shapes=[(N,)], output_dtypes=[mx.float32])[0]

if __name__ == "__main__":
    W = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
    x = mx.random.normal((K,)).astype(mx.float16)
    mx.eval(W, x)
    ref = W.astype(mx.float32) @ x.astype(mx.float32)
    for name, k in (("A", kA), ("B", kB)):
        err = mx.abs(gemv(k, W, x) - ref).max().item()
        assert err < 1e-3, (name, err)                  # correctness before speed
        print(f"layout {name}: max abs error vs fp32 reference {err:.2e}")
    print(f"mx.matmul (half) max abs error {mx.abs((W @ x).astype(mx.float32) - ref).max().item():.2e}")
    print("\nfloor at roof: %.0f us" % (BYTES / (bench.json.load(open(bench.MACHINE))["copy_gbps"] * 1e9) * 1e6))
    print("layout A vs mx.matmul : ", end=""); bench.paired(lambda: gemv(kA, W, x), lambda: W @ x, bytes_moved=BYTES)
    print("layout B vs mx.matmul : ", end=""); bench.paired(lambda: gemv(kB, W, x), lambda: W @ x, bytes_moved=BYTES)
    print("layout A vs layout B  : ", end=""); bench.paired(lambda: gemv(kA, W, x), lambda: gemv(kB, W, x), bytes_moved=BYTES)
