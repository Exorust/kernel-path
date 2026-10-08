"""Week 0, day 5, file 2: what is the kernel waiting on? Memory, math, or the launch.

    .venv/bin/python kernels/w00/day5_waiting_on.py

Nothing to fill in. Three experiments change one thing at a time and show whether the time follows it.
Reading: sessions/w00-d5.md.
"""
import mlx.core as mx
from helpers import run_parts, timed, today

# One float in, one float out, and ITERS turns of a small loop in between.
# One turn is one multiply and one add: 2 operations. ITERS = 1 is almost no math. ITERS = 4096 is a lot.
SOURCE = """
    uint i = thread_position_in_grid.x;
    float x = a[i];                                 // one load from memory
    for (uint k = 0; k < ITERS; ++k) {              // ITERS is a number fixed when the kernel is built
        x = x * 0.999f + 0.5f;                      // math only: x lives in a register, no memory is touched
    }
    out[i] = x;                                     // one store to memory
"""
kernel = mx.fast.metal_kernel(name="loop", input_names=["a"], output_names=["out"], source=SOURCE)


def measure(n, iters):
    a = mx.random.uniform(shape=(n,)); mx.eval(a)
    return timed(lambda: kernel(inputs=[a], template=[("ITERS", iters)], grid=(n, 1, 1), threadgroup=(32, 1, 1),
                                output_shapes=[(n,)], output_dtypes=[mx.float32])[0])[0] * 1e6


def table(rows):
    print(f"\n  {'elements':>12}  {'loop turns':>10}  {'bytes moved':>12}  {'operations':>15}  {'time':>10}   vs row above")
    times = []
    for n, iters in rows:
        t = measure(n, iters)
        step = f"{t / times[-1]:.1f} x" if times else ""
        times.append(t)
        print(f"  {n:>12,}  {iters:>10,}  {8 * n / 1e6:>9,.3f} MB  {2 * n * iters:>15,}  {t:>7,.0f} us   {step}")
    return times


def part_a():
    print("  The test: change one thing by 4 times. If the time clearly follows it, that thing is what the kernel waits on.")
    print("\n  Experiment 1: 4 times the bytes per row, almost no math")
    t = table([(4_000_000, 1), (16_000_000, 1), (64_000_000, 1)])
    print(f"  The time follows the bytes: this kernel waits on MEMORY. It is under 4x because every row pays the same launch cost, about {LAUNCH:,.0f} us.")
    print("\n  Experiment 2: the same 8 MB, 4 times the math per row")
    t = table([(1_000_000, 256), (1_000_000, 1024), (1_000_000, 4096)])
    global RATE
    RATE = 2 * 1_000_000 * 4096 / ((t[2] - LAUNCH) / 1e6)
    print(f"  The time follows the math: this kernel waits on MATH. Math speed in this loop: about {RATE / 1e9:,.0f} billion operations per second.")
    print("\n  Experiment 3: a tiny array; 4 times the bytes, then 4 times the math")
    t = table([(1_000, 1), (4_000, 1), (1_000, 4)])
    print(f"  The time does not move: this kernel waits on the LAUNCH. All three rows cost about one launch, {t[0]:,.0f} us here.")
    return "read the three tables"


def part_b():
    n, iters = 4_000_000, 256
    memory = 8 * n / 1e6 / R * 1000
    math = 2 * n * iters / RATE * 1e6
    print("  Before running a kernel, work out both times. The larger one is what it will wait on.")
    print(f"\n  4,000,000 elements, 256 loop turns each:")
    print(f"    memory time = bytes / memory speed      = 32 MB / {R:.0f} GB/s           = {memory:>7,.0f} us")
    print(f"    math time   = operations / math speed   = {2 * n * iters / 1e6:,.0f} M ops / {RATE / 1e9:,.0f} B ops/s = {math:>7,.0f} us")
    print(f"    waiting on: {'math' if math > memory else 'memory'}")
    print(f"    predicted   = launch + the larger      = {LAUNCH:,.0f} + {max(memory, math):,.0f}             = {LAUNCH + max(memory, math):>7,.0f} us")
    t = measure(n, iters)
    print(f"    measured                                                      = {t:>7,.0f} us     off by {abs(t - LAUNCH - max(memory, math)) / t * 100:.0f}%")
    return "read the prediction"


if __name__ == "__main__":
    R, LAUNCH = today()["roof"], today()["launch"]
    RATE = None
    print(f"Today: memory speed {R:.0f} GB/s, launch cost about {LAUNCH:,.0f} us.")
    run_parts([("A  three experiments", part_a),
               ("B  predict a time before it runs", part_b)])
