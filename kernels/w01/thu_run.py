"""Week 1 Thu runner: compiles a thu_*.metal body, checks it, times it against MLX.

    .venv/bin/python kernels/w01/thu_run.py vadd
    .venv/bin/python kernels/w01/thu_run.py reduce
"""
import ast, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import mlx.core as mx
import bench

N = 1 << 20
a, b = mx.random.normal((N,)), mx.random.normal((N,))
mx.eval(a, b)
# name: (inputs, output length, MLX reference, bytes moved)
CASES = {
    "vadd":   ({"a": a, "b": b}, N,       lambda: a + b,                      3 * N * 4),
    "reduce": ({"a": a},         N // 32, lambda: a.reshape(-1, 32).sum(axis=1), N * 4 + N // 32 * 4),
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    inputs, n_out, ref, nbytes = CASES[name]
    src = open(os.path.join(HERE, f"thu_{name}.metal")).read()
    m = re.search(r"^// grid=(\([\d, ]+\)) threadgroup=(\([\d, ]+\))", src, re.M)
    if not m:
        sys.exit(f"thu_{name}.metal: Blank 0 is not filled in (the '// grid=(...) threadgroup=(...)' line)")
    grid, tg = ast.literal_eval(m[1]), ast.literal_eval(m[2])
    k = mx.fast.metal_kernel(name=f"thu_{name}", input_names=list(inputs), output_names=["out"], source=src)
    run = lambda: k(inputs=list(inputs.values()), grid=grid, threadgroup=tg,
                    output_shapes=[(n_out,)], output_dtypes=[mx.float32])[0]
    err = mx.abs(run() - ref()).max().item()
    print(f"compiled. grid={grid} threadgroup={tg}. max abs error vs MLX {err:.2e}")
    assert err < 1e-3, "output does not match the MLX reference"   # correctness before speed
    print("yours vs MLX: ", end=""); bench.paired(run, ref, bytes_moved=nbytes)
