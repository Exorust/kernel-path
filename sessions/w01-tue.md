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
