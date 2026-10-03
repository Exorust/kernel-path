# w07-tue · How fast is one 8x8 multiply?

## The question

`simdgroup_matrix<float, 8, 8>` multiplies two 8 by 8 tiles in one instruction across the 32 lanes. If a kernel does nothing else, what throughput does it reach?

**Given.** One 8x8·8x8 multiply-accumulate is 512 multiplies and 512 adds = 1024 FLOPs. Each lane holds 2 of the 64 elements. The M5 peak for this path is on your `nax-gemm` page; use that number, do not guess one.

**Work out before running.**
1. Bytes of registers for three live tiles (A, B, C) per simdgroup, in fp32 and in bf16.
2. Using week 3's cliff, how many tiles a thread can hold before occupancy drops.
3. Predicted GFLOPS for a loop of 4096 multiply-accumulates per simdgroup with tiles kept in registers, as a percent of the peak on your page.
4. The same if the tiles are reloaded from device memory every iteration. Which bound takes over?

**The Ask must name:** both variants (register-resident and reloaded), fp32 and bf16, enough simdgroups to saturate 10 cores, GFLOPS computed from iterations times 1024, and `bench.paired` between the two variants.

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
