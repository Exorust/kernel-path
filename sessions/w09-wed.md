# w09-wed · Direct a Triton chunkwise kernel

## The question

A simplified Triton version of week 8's kernel: scalar gate, one chunk size. It will lose to FLA. By how much, and where?

**Given.** FLA's time from Tuesday. FlashQLA claims 2 to 3x over FLA on Hopper through warpgroup specialization. Your kernel has neither FLA's tuning nor that.

**Work out before running.**
1. Predicted ratio to FLA at T=4096.
2. The stage where you expect the largest gap, and the Hopper feature (TMA, wgmma, pipelining through num_stages) responsible.
3. What `tl.dot` should compile to on sm_90. What you will grep the PTX for to confirm it.
4. One thing your Mac kernel does that has no direct Triton equivalent, or the reverse.

**The Ask must name:** scalar gate, C=64, the four stages as separate Triton kernels, FLA as baseline, a PTX grep for `wgmma` and `cp.async`, and parity against FLA's `naive.py`.

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
