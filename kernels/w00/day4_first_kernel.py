"""Week 0, day 4: the smallest kernels, and the function MLX writes around them.

    .venv/bin/python kernels/w00/day4_first_kernel.py

Three finished kernels. Read each one, run the file, and compare the printed output with the code.
Set VERBOSE = True to see the full Metal function that MLX generates around part A.
Reading: sessions/w00-d4.md. Cheat sheet: docs/metal-cheatsheet.md.
"""
import mlx.core as mx
from helpers import run_parts, show

N = 8                           # a small array, so the whole result fits on one line
VERBOSE = False                 # set to True to print the function MLX writes around the body of part A

a = mx.arange(N).astype(mx.float32)                 # 0, 1, 2, ... 7
b = 100 * mx.ones((N,))                             # 100, 100, ... 100

# ------------------------------------------------------------------ Part A: a finished example
# On a CPU you would write:      for i in range(N): out[i] = 2 * a[i]
# On a GPU the loop is gone. The loop body is the kernel, and N threads each run it once.
SOURCE_A = """
    uint i = thread_position_in_grid.x;     // which element is mine? every thread gets a different i
    out[i] = 2.0f * a[i];                   // read my element of a, double it, write my element of out
"""


def part_a():
    kernel = mx.fast.metal_kernel(
        name="double",
        input_names=["a"],                  # the body may read through a pointer called a
        output_names=["out"],               # the body may write through a pointer called out
        source=SOURCE_A,                    # the body only: MLX writes the function around it
    )
    out = kernel(
        inputs=[a],                         # the array that a points to
        grid=(N, 1, 1),                     # how many threads in total: one per element
        threadgroup=(N, 1, 1),              # how the threads are bundled (day 2)
        output_shapes=[(N,)],               # how big out is
        output_dtypes=[mx.float32],         # what type out holds
        verbose=VERBOSE,
    )[0]
    show("a  ", a)
    show("out", out)
    return "matches" if mx.array_equal(out, 2 * a).item() else "out is not 2 * a"


# ------------------------------------------------------------------ Part B: out[i] = a[i] + 1
SOURCE_B = """
    uint i = thread_position_in_grid.x;    // which element is mine? The same line as in part A.
    out[i] = a[i] + 1.0f;                  // my element of a, plus one. A float literal ends in f: 1.0f
"""


def part_b():
    kernel = mx.fast.metal_kernel(name="add_one", input_names=["a"], output_names=["out"], source=SOURCE_B)
    out = kernel(inputs=[a], grid=(N, 1, 1), threadgroup=(N, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    show("a  ", a)
    show("out", out)
    return "matches" if mx.array_equal(out, a + 1).item() else "out is not a + 1: compare the two rows above"


# ------------------------------------------------------------------ Part C: out[i] = a[i] + b[i]
GRID_C = (N, 1, 1)                      # how many threads this launch needs: one per element

SOURCE_C = """
    uint i = thread_position_in_grid.x;    // which element is mine?
    float x = a[i];                         // my element of a
    float y = b[i];                         // my element of b
    out[i] = x + y;                         // their sum
"""


def part_c():
    kernel = mx.fast.metal_kernel(name="add_two", input_names=["a", "b"], output_names=["out"], source=SOURCE_C)
    out = kernel(inputs=[a, b], grid=GRID_C, threadgroup=(N, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    show("a  ", a)
    show("b  ", b)
    show("out", out)
    return "matches" if mx.array_equal(out, a + b).item() else "out is not a + b: compare the rows above"


if __name__ == "__main__":
    run_parts([("A  double every element (finished example)", part_a),
               ("B  add one", part_b),
               ("C  add two arrays", part_c)])
