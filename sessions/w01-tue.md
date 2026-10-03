# w01-tue · How wide should a threadgroup be?

## The question

You will time a kernel that only copies an array, and sweep its launch shape. Which launch shape reaches the highest GB/s, and how far below the best is the worst?

**Given.** Array: 2^26 fp32 values = 256 MB. A copy reads it once and writes it once, so 512 MB moves. This machine's copy roof is 93 GB/s (`machine.json`), so the floor is about 5.5 ms. The M5 has 10 GPU cores. A simdgroup is 32 threads. A threadgroup holds at most 1024 threads. Apple's rule: 1K to 2K concurrent threads per core saturates it; your glossary says the ALUs saturate near 24 resident simdgroups (768 threads).

**Sweep.** Threadgroup width in {32, 64, 128, 256, 512, 1024} times elements copied per thread in {1, 2, 4, 8}. That is 24 cells.

**Write down before running.**
1. Which single cell you expect to be fastest, and its GB/s.
2. Which cell you expect to be slowest, and why (too few threads in flight? too many launches? uneven spread across cores?).
3. Whether elements-per-thread or threadgroup width matters more, in one sentence with the reason.
4. Whether any cell can exceed 93 GB/s, and what it would mean if one did.

**The Ask must name:** the array size, both sweep axes, `bench.paired` against `mx.array + 0` as baseline, `bytes_moved=512 MB`, and a printed 6 by 4 table of GB/s.

## How to break this down

The kernel is boring on purpose: copy 256 MB, 24 launch shapes, see which is fastest. Only the way the work is cut up changes.

**The two knobs.**
- *Elements per thread* changes how many threads exist: 2^26 elements at 1 per thread is 67 million threads; at 8 per thread, 8.4 million. Fewer threads, same bytes.
- *Threadgroup width* changes how threads are bundled, not how many there are. It decides how many bundles go out to the 10 cores and how evenly.

**Ask three things of any cell, in this order.**
1. Are enough threads in flight? The rule is 10K to 20K for 10 cores. The smallest count in this sweep is 8.4 million, so every cell passes. This cannot be what separates fast from slow.
2. How much overhead per byte? Every thread costs something to start, however much it copies. A thread copying 8 elements spreads that cost over 8 times the bytes.
3. Does the bundling hurt? A copy uses no threadgroup memory, so threads in a group share nothing. Apple's advice: the smallest multiple of 32 that fits, because large groups spread unevenly.

**One cell worked.** Width 256, 1 element per thread: 67 million threads, 262 thousand threadgroups, 8 bytes per thread (4 read, 4 written). Floor: 512 MB / 93 GB/s = 5.5 ms. If thread start cost is small this lands near the floor; if not, it is among the slowest, because it has the most threads per byte.

**Then the four answers follow.** Fastest cell: from question 2. Slowest: from 2 and 3 together. Which knob matters more: whether question 3 mattered at all. Can a cell beat 93 GB/s: the roof came from a quick probe, and the glossary lists a base M5 near 153 GB/s, so a cell above 93 means the probe was low. That is a finding, not an error.

## Before (written by me, before any code exists)
- The [three questions](https://metalworking.vercel.app/war-stories/three-questions/): can I delete work? unlock an existing fast path? cut dispatch/sync overhead?
- Bound (memory / compute / latency / launch) and why:
- Plan (who owns what: thread, simdgroup, threadgroup; what lives in registers vs threadgroup memory):
- Prediction, with the arithmetic shown:
- What would make me wrong:

## Ask (what I told Claude to write, one paragraph)

## After
- Measured (time, x vs baseline, CV %, roofline %):
- Gap between prediction and measurement, explained in my words:
- One thing I would try next:
