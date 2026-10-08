"""Week 0, day 5, file 3: the sum of every row of a matrix. Predict, run, compare, then change the shape.

    .venv/bin/python kernels/w00/day5_rowsum.py

Nothing to fill in. This is week 1 Wednesday's kernel without the multiply.
Reading: sessions/w00-d5.md. Cheat sheet: docs/metal-cheatsheet.md, sections 5 and 8.
"""
import contextlib, io, re
import mlx.core as mx
from helpers import run_parts, timed, today
import bench                    # found through helpers, which puts the repo root on the path

ROWS, COLS = 2_000, 4_000       # W has 2,000 rows of 4,000 floats = 32 MB. out[r] is the sum of row r.

# The plan: one simdgroup (32 threads) per row. The grid is 32 threads across and ROWS down, so
# thread_position_in_grid.x is the lane (0 to 31) and thread_position_in_grid.y is the row.
# Lane 0 adds columns 0, 32, 64, ...  Lane 1 adds columns 1, 33, 65, ...  Then simd_sum adds the 32 partial sums.
SOURCE = """
    uint lane = thread_position_in_grid.x;          // my place across the row, 0 to 31
    uint row  = thread_position_in_grid.y;          // which row my simdgroup owns

    float acc = 0.0f;                               // my partial sum. It lives in a register: private and fast.
    for (uint c = lane; c < COLS; c += 32) {        // my first column is my lane; then every 32nd column
        acc += W[row * COLS + c];                   // element (row, c). W is flat: row r starts at r * COLS.
    }

    float total = simd_sum(acc);                    // add the 32 partial sums; every lane gets the total
    if (lane == 0) out[row] = total;                // exactly one thread stores the row's sum
"""
kernel = mx.fast.metal_kernel(name="rowsum", input_names=["W"], output_names=["out"], source=SOURCE)


def rowsum(W, rows, cols):
    return kernel(inputs=[W], template=[("COLS", cols)], grid=(32, rows, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows,)], output_dtypes=[mx.float32])[0]


def correct(W, rows, cols):
    want = W.sum(axis=1)
    return (mx.abs(rowsum(W, rows, cols) - want) / mx.abs(want)).max().item() < 1e-4


def part_a():
    memory = 4 * ROWS * COLS / 1e6 / R * 1000
    math = ROWS * COLS / RATE * 1e6
    print(f"  bytes read   = {ROWS:,} x {COLS:,} x 4 bytes = 32 MB")
    print(f"  memory time  = 32 MB / {R:.0f} GB/s                  = {memory:>6,.0f} us")
    print(f"  math time    = {ROWS * COLS / 1e6:.0f} M adds / {RATE / 1e9:,.0f} B ops/s      = {math:>6,.0f} us")
    print(f"  waiting on   : memory, by a wide margin")
    print(f"  predicted    = launch + memory time = {LAUNCH:,.0f} + {memory:,.0f}   = {LAUNCH + memory:>6,.0f} us")
    return "the prediction, before any run"


def part_b():
    W = mx.random.uniform(shape=(ROWS, COLS)); mx.eval(W)
    if not correct(W, ROWS, COLS):
        return "the sums do not match MLX's W.sum(axis=1)"
    print("  Every row sum matches MLX's own.\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                # bench.paired prints one line; catch it so it can be explained
        bench.paired(lambda: rowsum(W, ROWS, COLS), lambda: W.sum(axis=1), bytes_moved=4 * ROWS * COLS)
    line = buf.getvalue().strip()
    m = re.search(r"speedup x([\d.]+)\s+candidate ([\d.]+) us\s+CV ([\d.]+)%\s+roofline (\d+)% of (\d+)", line)
    x, cand, cv, pct, peak = float(m[1]), float(m[2]), float(m[3]), int(m[4]), int(m[5])
    predicted = LAUNCH + 4 * ROWS * COLS / 1e6 / R * 1000
    print(f"  predicted {predicted:,.0f} us     measured {cand:,.0f} us     off by {abs(cand - predicted) / cand * 100:.0f}%\n")
    print("  This kernel against MLX's own sum. Every later session prints a line like this one:\n")
    print("      " + line + "\n")
    cmp = f"this kernel is {x:.2f} times as fast as MLX's" if x >= 1 else f"this kernel takes {1 / x:.2f} times as long as MLX's"
    print(f"      speedup x{x:.2f}       MLX's time divided by this kernel's time: {cmp}")
    print(f"      candidate {cand:.0f} us   this kernel's time for one launch")
    print(f"      CV {cv:.1f}%            how much that time varied between runs. Above 8% the number is noise: close apps, rerun.")
    print(f"      roofline {pct}%       bytes / time, as a share of the {peak} GB/s memory speed.")
    print(f"                         A launch costs about {LAUNCH:,.0f} us before any byte moves, so a 32 MB kernel cannot reach 100%.")
    return "read the line"


def part_c():
    print("  Four matrices, all 32 MB. One simdgroup per row, so the number of rows sets the number of threads.\n")
    print(f"  {'rows':>8} x {'columns':>10}  {'threads':>10}  {'this kernel':>12}  {'MLX own sum':>12}")
    times = {}
    for rows in (200_000, 2_000, 20, 2):
        cols = ROWS * COLS // rows
        W = mx.random.uniform(shape=(rows, cols)); mx.eval(W)
        if not correct(W, rows, cols):
            return f"the sums are wrong for {rows} rows"
        times[rows] = timed(lambda: rowsum(W, rows, cols))[0] * 1e6
        ref = timed(lambda: W.sum(axis=1))[0] * 1e6
        print(f"  {rows:>8,} x {cols:>10,}  {32 * rows:>10,}  {times[rows]:>9,.0f} us  {ref:>9,.0f} us")
    slowest = max(times, key=times.get)
    print(f"\n  Same 32 MB and the same adds, yet {times[slowest] / min(times.values()):,.0f} times slower with {32 * slowest} threads.")
    print("  With 2 rows the kernel is short of threads: 2 simdgroups can keep at most 2 of the 10 cores busy, and each")
    print("  one waits for every load before it can issue the next. MLX's own sum does not use one simdgroup per row, so it is unaffected.")
    return "read the table"


if __name__ == "__main__":
    R, LAUNCH, RATE = today()["roof"], today()["launch"], today()["rate"]
    print(f"Today: memory speed {R:.0f} GB/s, launch cost about {LAUNCH:,.0f} us, math speed about {RATE / 1e9:,.0f} billion operations per second.")
    run_parts([("A  the prediction", part_a),
               ("B  the kernel, measured", part_b),
               ("C  same bytes, different shape", part_c)])
