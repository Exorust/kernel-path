"""Week 0, day 3: how the GPU runs a kernel. Threads come in bundles.

    .venv/bin/python kernels/w00/day3_threads.py

Five finished kernels. Parts A and D have a setting to change: change it and run again.
Reading: sessions/w00-d3.md. Cheat sheet: docs/metal-cheatsheet.md, sections 2, 5 and 7.
"""
import mlx.core as mx
from helpers import run_parts, show, show_simdgroups

N = 128                         # 128 threads: enough for 4 simdgroups of 32

a = mx.arange(N).astype(mx.float32)                 # 0, 1, 2, ... 127

# ------------------------------------------------------------------ Part A: ask every thread who it is
# Two settings to change. Run the file again after each change and read the four lines it prints.
WHO = "thread_index_in_simdgroup"       # also try: simdgroup_index_in_threadgroup
                                        #           thread_index_in_threadgroup
                                        #           threadgroup_position_in_grid.x
WIDTH = 32                              # threads per threadgroup. Also try 64 and 128.

SOURCE_A = f"""
    uint i = thread_position_in_grid.x;
    out[i] = float({WHO});                  // every thread writes down its answer to the question
"""


def part_a():
    kernel = mx.fast.metal_kernel(name="who", input_names=["a"], output_names=["out"], source=SOURCE_A)
    out = kernel(inputs=[a], grid=(N, 1, 1), threadgroup=(WIDTH, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    print(f"  question: {WHO}      threadgroup width: {WIDTH}\n")
    show_simdgroups(out)
    return "ran: change WHO or WIDTH and run again"


# ------------------------------------------------------------------ Part B: the 32 lanes add up their values
SOURCE_B = """
    uint i = thread_position_in_grid.x;
    float mine  = a[i];                     // each thread holds one value
    float total = simd_sum(mine);           // one call: the sum of that value over the 32 threads of my simdgroup
    out[i] = total;                         // every thread writes down the total it got
"""


def part_b():
    kernel = mx.fast.metal_kernel(name="lane_sum", input_names=["a"], output_names=["out"], source=SOURCE_B)
    out = kernel(inputs=[a], grid=(N, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    print("  a is 0, 1, 2, ... 127. After simd_sum, this is what each thread holds:\n")
    show_simdgroups(out)
    ok = mx.array_equal(out, mx.repeat(a.reshape(-1, 32).sum(axis=1), 32)).item()
    return "all 32 threads of a simdgroup hold the same total" if ok else "unexpected output"


# ------------------------------------------------------------------ Part C: one sum per group of 32
# out has only N / 32 = 4 elements now: out[g] is the sum of a[32g] to a[32g + 31].
SOURCE_C = """
    uint i    = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;                       // my place in my simdgroup, 0 to 31
    uint g    = i / 32;                       // which group of 32 am I in? i counts threads, 32 per group.

    float total = simd_sum(a[i]);                     // add a[i] across the 32 lanes, as in part B

    if (lane == 0) out[g] = total;               // all 32 lanes hold the same total. Let exactly one store it.
"""


def part_c():
    kernel = mx.fast.metal_kernel(name="group_sum", input_names=["a"], output_names=["out"], source=SOURCE_C)
    out = kernel(inputs=[a], grid=(N, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(N // 32,)], output_dtypes=[mx.float32])[0]
    want = a.reshape(-1, 32).sum(axis=1)
    show("out ", out)
    show("want", want)
    return "matches" if mx.array_equal(out, want).item() else "out does not match want"


# ------------------------------------------------------------------ Part D: a scratch memory one threadgroup shares
SCRATCH_WIDTH = 32                      # threads per threadgroup. Also try 64.

SOURCE_D = """
    threadgroup float scratch[TG];          // a small memory that the threads of ONE threadgroup share
    uint t = thread_index_in_threadgroup;   // my place in my threadgroup
    uint i = thread_position_in_grid.x;

    scratch[t] = a[i];                                      // every thread puts its value in the scratch memory
    threadgroup_barrier(mem_flags::mem_threadgroup);        // wait here until every thread of the group has done so
    out[i] = scratch[TG - 1 - t];                           // then take the value a DIFFERENT thread put there
"""


def part_d():
    kernel = mx.fast.metal_kernel(name="scratch", input_names=["a"], output_names=["out"], source=SOURCE_D)
    out = kernel(inputs=[a], template=[("TG", SCRATCH_WIDTH)], grid=(N, 1, 1), threadgroup=(SCRATCH_WIDTH, 1, 1),
                 output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    print(f"  threadgroup width: {SCRATCH_WIDTH}. a is 0, 1, 2, ... 127. out is:\n")
    show_simdgroups(out)
    return "ran: change SCRATCH_WIDTH and run again"


# ------------------------------------------------------------------ Part E: a grid with two dimensions
# 4 threads across and 3 down: 12 threads. Each one knows its position across (.x) and down (.y).
SOURCE_E = """
    uint x = thread_position_in_grid.x;     // 0 to 3: my position across
    uint y = thread_position_in_grid.y;     // 0 to 2: my position down
    out[y * 4 + x] = float(10 * y + x);     // write "down, across" as a two-digit number
"""


def part_e():
    kernel = mx.fast.metal_kernel(name="grid2d", input_names=["a"], output_names=["out"], source=SOURCE_E)
    out = kernel(inputs=[a], grid=(4, 3, 1), threadgroup=(4, 1, 1), output_shapes=[(12,)], output_dtypes=[mx.float32])[0]
    print("  grid = (4, 3, 1). Each number is one thread: tens digit = y (down), ones digit = x (across).\n")
    show("out", out, per_row=4)
    return "ran"


if __name__ == "__main__":
    run_parts([("A  who am I? (finished example)", part_a),
               ("B  simd_sum (finished example)", part_b),
               ("C  one sum per group of 32", part_c),
               ("D  scratch memory (finished example)", part_d),
               ("E  a grid with two dimensions (finished)", part_e)])
