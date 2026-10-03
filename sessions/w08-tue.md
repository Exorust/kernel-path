# w08-tue · Barrier cost vs chunk size

## The question

Carrying state between chunks needs every simdgroup in a threadgroup to agree on S before the next chunk starts, which means a threadgroup barrier. How much does one barrier cost, and at what chunk size does it stop mattering?

**Given.** T=4096. Chunk sizes C in {16, 32, 64, 128} give 256, 128, 64, 32 barriers. Work per chunk grows roughly as C²·D for the intra-chunk products. v4 of the workbench kernel was rejected because of barriers.

**Work out before running.**
1. Barriers per sequence at each C.
2. If one barrier costs b microseconds and intra-chunk work costs w(C), the total is (T/C)·(b + w(C)). Sketch which term wins at small C and at large C.
3. Predicted C where barrier time falls under 10 percent of the total.
4. What a larger C costs you elsewhere (registers? threadgroup memory against the 32 KB limit? the triangular solve?).

**The Ask must name:** a kernel that does a fixed dummy amount of tile work per chunk followed by one `threadgroup_barrier`, the same kernel without the barrier, the four chunk sizes, and the per-barrier cost derived from the difference.

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
