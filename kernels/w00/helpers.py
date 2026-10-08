"""Shared by the week 0 files, so that each lesson file can stay about its kernel.

You do not need to read this file to do the lessons.
"""
import contextlib, io, json, os, re, sys, time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import mlx.core as mx
import bench

BLANK = "____"
worst_drift = 0.0               # the least steady timing seen in this run


def blanks(*things):
    """How many blanks are still unfilled: ____ inside a string, or None anywhere else."""
    n = 0
    for t in things:
        if isinstance(t, str):
            n += t.count(BLANK)
        elif isinstance(t, (tuple, list)):
            n += blanks(*t)
        elif t is None:
            n += 1
    return n


def run_parts(parts):
    """Run each (title, function) in turn. A function prints what it likes and returns a one-line status."""
    results = []
    for title, fn in parts:
        print(f"\n=== {title} ===")
        try:
            status = fn()
        except RuntimeError as e:
            if "Unable to build metal library" not in str(e):
                raise
            print("  The Metal compiler rejected this kernel body. Its message:\n")
            for line in str(e).splitlines()[1:]:
                if line.strip():
                    print("      " + re.sub(r"^mlx/backend/metal/kernels/utils\.h:\d+:\d+: ", "", line))
            print("\n  How to read it: the first line names the problem, the next line is your line of code,")
            print("  and the ^ marks the exact place. Fix that line and run the file again.")
            status = "does not compile yet (the compiler's message is above)"
        results.append((title, status))
    print("\n" + "-" * 78)
    for title, status in results:
        print(f"{title:<46} {status}")
    if worst_drift > 0.2:
        print(f"\nNote: one timing moved by {worst_drift * 100:.0f}% while it was being measured. Other apps were using the machine.")
        print("The patterns still hold. For steadier numbers, close heavy apps and run again.")


def show(label, x, per_row=16):
    """Print a small array as rows of numbers."""
    vals = [f"{v:g}" for v in x.tolist()]
    w = max(4, max(len(v) for v in vals))               # at least 4 wide, so rows of different arrays line up
    for r in range(0, len(vals), per_row):
        print(f"  {label if r == 0 else ' ' * len(label)}  " + " ".join(v.rjust(w) for v in vals[r:r + per_row]))


def show_simdgroups(x):
    """Print an array 32 values at a time (one simdgroup per line), in a short form when there is a pattern."""
    vals = [int(v) if float(v).is_integer() else v for v in x.tolist()]
    for g in range(0, len(vals), 32):
        row = vals[g:g + 32]
        if all(v == row[0] for v in row):
            text = f"all 32 threads say {row[0]}"
        elif all(row[j + 1] - row[j] == 1 for j in range(31)):
            text = f"{row[0]}, {row[1]}, {row[2]}, ... up to {row[-1]}"
        elif all(row[j + 1] - row[j] == -1 for j in range(31)):
            text = f"{row[0]}, {row[1]}, {row[2]}, ... down to {row[-1]}"
        else:
            text = " ".join(str(v) for v in row)
        print(f"  threads {g:>3} to {g + 31:>3}:   {text}")


def timed(fn, least=15, budget=0.25):
    """Seconds for one launch of fn, and how unsteady the machine was while measuring.

    Each launch is timed by itself and the median is kept, so one slow launch (another app
    taking the GPU for a moment) does not move the answer. The second value compares the
    first half of the launches with the second half: near 0 means the machine held still.
    """
    mx.eval(fn()); mx.eval(fn())                        # the first launches compile the kernel and warm up
    ts, start = [], time.perf_counter()
    while len(ts) < least or (time.perf_counter() - start < budget and len(ts) < 300):
        t = time.perf_counter()
        mx.eval(fn())                                   # MLX is lazy: nothing runs until the result is asked for
        ts.append(time.perf_counter() - t)
    half = len(ts) // 2
    med = sorted(ts)[half]
    drift = abs(sorted(ts[:half])[half // 2] - sorted(ts[half:])[(len(ts) - half) // 2]) / med
    global worst_drift
    worst_drift = max(worst_drift, drift)
    return med, drift


def roof():
    """This machine's memory speed in GB/s. Measured once a day and kept in machine.json."""
    try:
        m = json.load(open(bench.MACHINE))
        if m.get("when") == time.strftime("%Y-%m-%d"):
            return m["copy_gbps"]
    except (OSError, ValueError, KeyError):
        pass
    with contextlib.redirect_stdout(io.StringIO()):
        return bench.probe()


_double = mx.fast.metal_kernel(name="w00_double", input_names=["a"], output_names=["out"],
                               source="uint i = thread_position_in_grid.x; out[i] = 2.0f * a[i];")
_loop = mx.fast.metal_kernel(name="w00_loop", input_names=["a"], output_names=["out"], source="""
    uint i = thread_position_in_grid.x; float x = a[i];
    for (uint k = 0; k < ITERS; ++k) { x = x * 0.999f + 0.5f; }
    out[i] = x;""")


def _tiny():
    a = mx.random.uniform(shape=(1000,)); mx.eval(a)
    return lambda: _double(inputs=[a], grid=(1000, 1, 1), threadgroup=(32, 1, 1),
                           output_shapes=[(1000,)], output_dtypes=[mx.float32])[0]


def launch_cost():
    """Seconds one launch costs before any real work: a kernel on 1,000 elements, launched back to back."""
    return timed(_tiny(), least=100)[0]


def launch_after_pause(pause):
    """The same launch, but with the GPU left idle for `pause` seconds before each one."""
    fn, ts = _tiny(), []
    mx.eval(fn())
    for _ in range(25):
        time.sleep(pause)
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def math_rate(launch):
    """Operations per second this GPU reaches in a simple multiply-add loop (day 5 file 2 measures the same thing)."""
    n, iters = 1_000_000, 1024
    a = mx.random.uniform(shape=(n,)); mx.eval(a)
    t = timed(lambda: _loop(inputs=[a], template=[("ITERS", iters)], grid=(n, 1, 1), threadgroup=(32, 1, 1),
                            output_shapes=[(n,)], output_dtypes=[mx.float32])[0])[0]
    return 2 * n * iters / (t - launch)


TODAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".today.json")


def today():
    """This machine's three numbers: roof (GB/s), launch (us), rate (operations per second).

    Measured on the first run of the day and then reused, so that the arithmetic you do with
    them can be checked exactly. Delete kernels/w00/.today.json to measure again.
    """
    day = time.strftime("%Y-%m-%d")
    try:
        m = json.load(open(TODAY))
        if m.get("when") == day:
            return m
    except (OSError, ValueError):
        pass
    launch = launch_cost()
    m = {"when": day, "roof": roof(), "launch": launch * 1e6, "rate": math_rate(launch)}
    json.dump(m, open(TODAY, "w"))
    return m


def check(label, mine, right, unit, tol=0.05, source="the arithmetic gives"):
    """Print one line comparing a number you filled in with the right one. True if it is close enough."""
    ok = abs(mine - right) <= tol * abs(right)
    note = "ok"
    if not ok:
        note = "check the arithmetic"
        ratio = mine / right if right else 0.0
        for factor, hint in ((1000, "1,000 times too big: a unit slip?"), (0.001, "1,000 times too small: a unit slip?"),
                             (2, "double: was something counted twice?"), (0.5, "half: both the read and the write move bytes")):
            if abs(ratio / factor - 1) < 0.06:
                note = hint
    print(f"  {label:<30} you wrote {mine:>10,.4g} {unit:<3} {source} {right:>10,.4g} {unit:<3}  {note}")
    return ok
