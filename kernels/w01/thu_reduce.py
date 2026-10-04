"""Week 1 Thu, file 2: out[g] = sum of a[32g .. 32g+31]. a is fp32 with 2^20 elements, out has 2^15.
One simdgroup (32 threads) per output element.

    .venv/bin/python kernels/w01/thu_reduce.py

Replace every ____ below. The file runs at any time and says how many blanks are left.
Spec: sessions/w01-thu.md. Cheat sheet: docs/metal-cheatsheet.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import mlx.core as mx
import bench

____ = None                                         # the blank marker, so this file runs before you fill anything
N = 1 << 20
G = N // 32                                         # number of output elements

# Blank 0: one simdgroup per output element. Cheat sheet section 8.
GRID        = (____, ____, 1)
THREADGROUP = (____, 1, 1)

SOURCE = """
    uint lane = ____;           // Blank 1: 0 to 31, this thread's place in its simdgroup
    uint g    = ____;           // Blank 2: which output element this simdgroup owns

    float v = a[____];          // Blank 3: the one input value that belongs to this lane of group g

    float total = ____;         // Blank 4: add v across the 32 lanes. Cheat sheet section 5.

    if (____) out[g] = total;   // Blank 5: all 32 lanes hold the same total. Let exactly one store it.
"""

if __name__ == "__main__":
    left = SOURCE.count("____") + (GRID + THREADGROUP).count(None)
    if left:
        sys.exit(f"{left} blanks left to fill in {os.path.basename(__file__)}")
    a = mx.random.normal((N,))
    mx.eval(a)
    ref = lambda: a.reshape(G, 32).sum(axis=1)
    k = mx.fast.metal_kernel(name="thu_reduce", input_names=["a"], output_names=["out"], source=SOURCE)
    reduce = lambda: k(inputs=[a], grid=GRID, threadgroup=THREADGROUP,
                       output_shapes=[(G,)], output_dtypes=[mx.float32])[0]
    err = mx.abs(reduce() - ref()).max().item()
    print(f"compiled. max abs error vs MLX sum: {err:.2e}")
    assert err < 1e-3, "output does not match the MLX sum"   # correctness before speed
    print("yours vs MLX sum: ", end=""); bench.paired(reduce, ref, bytes_moved=N * 4 + G * 4)
