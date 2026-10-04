"""Week 1 Thu, file 1: out = a + b. fp32, 2^20 elements, one element per thread.

    .venv/bin/python kernels/w01/thu_vadd.py

Replace every ____ below. The file runs at any time and says how many blanks are left.
Spec: sessions/w01-thu.md. Cheat sheet: docs/metal-cheatsheet.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import mlx.core as mx
import bench

____ = None                                         # the blank marker, so this file runs before you fill anything
N = 1 << 20

# Blank 0: grid = how many threads in total. threadgroup = how they are bundled. Cheat sheet section 4.
GRID        = (____, 1, 1)
THREADGROUP = (____, 1, 1)

SOURCE = """
    uint i = ____;              // Blank 1: which element this thread owns. Cheat sheet section 2.

    float x = ____;             // Blank 2: load this thread's element of a
    float y = ____;             // Blank 3: load this thread's element of b

    out[____] = ____;           // Blank 4: where the result goes, and what it is
"""

if __name__ == "__main__":
    left = SOURCE.count("____") + (GRID + THREADGROUP).count(None)
    if left:
        sys.exit(f"{left} blanks left to fill in {os.path.basename(__file__)}")
    a, b = mx.random.normal((N,)), mx.random.normal((N,))
    mx.eval(a, b)
    k = mx.fast.metal_kernel(name="thu_vadd", input_names=["a", "b"], output_names=["out"], source=SOURCE)
    vadd = lambda: k(inputs=[a, b], grid=GRID, threadgroup=THREADGROUP,
                     output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    err = mx.abs(vadd() - (a + b)).max().item()
    print(f"compiled. max abs error vs a + b: {err:.2e}")
    assert err < 1e-3, "output does not match a + b"    # correctness before speed
    print("yours vs mx.add: ", end=""); bench.paired(vadd, lambda: a + b, bytes_moved=3 * N * 4)
