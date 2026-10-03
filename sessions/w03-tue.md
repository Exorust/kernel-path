# w03-tue · Where is the register cliff?

## The question

A kernel that keeps more live values per thread uses more of the core's register file, so fewer simdgroups fit at once. At how many live floats per thread does the kernel slow down?

**Given.** Register file per core: about 208 KB (measured on M1/M2, the M5 number is not published). ALUs saturate near 24 resident simdgroups = 768 threads. fp32 is 4 bytes.

**Work out before running.**
1. Registers available per thread at 768 resident threads: 208 KB / 768. Convert to a count of fp32 values.
2. From that, the live-float count where occupancy must start to drop. Is it below 32, between 32 and 64, between 64 and 128, or above 128?
3. What you expect past the cliff: a gradual slope or a step? Why?
4. Whether using bf16 values instead of fp32 would move the cliff, and to where.

**Sweep.** The week 1 copy kernel, modified so each thread first loads N values into a local array, then writes them: N in {8, 16, 32, 64, 128}. Same total bytes for every N.

**The Ask must name:** equal total bytes across N, the local array declared so the compiler keeps it in registers, `bench.paired` against the N=8 version, and a printed table of GB/s by N.

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
