"""Week 0, day 5, file 1: how long should a kernel take, and what a launch costs.

    .venv/bin/python kernels/w00/day5_launch_cost.py

Nothing to fill in. The file works out the arithmetic, measures, and prints both side by side.
Reading: sessions/w00-d5.md.
"""
import mlx.core as mx
from helpers import run_parts, timed, today, launch_after_pause

SOURCE = """
    uint i = thread_position_in_grid.x;
    out[i] = 2.0f * a[i];                   // read one float, write one float
"""
kernel = mx.fast.metal_kernel(name="double", input_names=["a"], output_names=["out"], source=SOURCE)


def double(a):
    n = a.size
    return kernel(inputs=[a], grid=(n, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0]


def measure(n):
    """Microseconds for this kernel on n elements, and for MLX's own a * 2 on the same array at the same moment."""
    a = mx.random.uniform(shape=(n,)); mx.eval(a)
    return timed(lambda: double(a))[0] * 1e6, timed(lambda: a * 2)[0] * 1e6


def part_a():
    print("  Rule 1: time = bytes moved / memory speed.   A float is 4 bytes; one read and one write per element.")
    print(f"  Memory speed today: {R:.0f} GB/s.   MB divided by GB/s gives milliseconds; times 1,000 gives microseconds (us).\n")
    print(f"  {'elements':>12}  {'bytes moved':>12}  {'rule 1 says':>12}  {'measured':>11}  {'MLX own a * 2':>14}   measured / rule")
    for n in (16_000_000, 64_000_000, 1_000):
        mb = 8 * n / 1e6
        rule = mb / R * 1000
        t, mlx_t = measure(n)
        shown = f"{rule:,.2f}" if rule < 10 else f"{rule:,.0f}"
        print(f"  {n:>12,}  {mb:>9,.3f} MB  {shown:>9} us  {t:>8,.0f} us  {mlx_t:>11,.0f} us   {t / rule:>10,.1f} x")
    print("\n  The two big arrays land near the rule. The tiny one is thousands of times off: that time is not memory.")
    return "read the table"


def part_b():
    launch = measure(1_000)[0]
    print(f"  A launch costs about {launch:,.0f} us before any byte moves. MLX's own a * 2 pays the same.")
    print("  Rule 2: time = launch cost + bytes moved / memory speed.\n")
    n = 4_000_000
    memory = 8 * n / 1e6 / R * 1000
    t = measure(n)[0]
    print(f"  4,000,000 elements, 32 MB moved:")
    print(f"    rule 1, memory time alone                  {memory:>7,.0f} us")
    print(f"    rule 2, launch cost + memory time          {launch + memory:>7,.0f} us")
    print(f"    measured                                   {t:>7,.0f} us     rule 2 is off by {abs(t - launch - memory) / t * 100:.0f}%")
    print(f"\n  Break-even: a kernel that moves {launch * R / 1000:,.0f} MB spends as long on the launch as on memory.")
    print("  Below that size, the launch is most of the time.\n")
    print("  The launch cost is not a constant:")
    print(f"    launches one right after another          {launch:>5,.0f} us each")
    print(f"    the same launch after a 20 ms pause       {launch_after_pause(0.02) * 1e6:>5,.0f} us")
    print("  A GPU that has been idle takes longer to start. Every timing in this repo launches back to back.")
    return "read the numbers"


if __name__ == "__main__":
    R = today()["roof"]
    run_parts([("A  rule 1: bytes / memory speed", part_a),
               ("B  rule 2: add the launch cost", part_b)])
