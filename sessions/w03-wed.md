# w03-wed · Batched qmv at M=1, 4, 8

## The question

Decode with M sequences at once means M activation vectors against the same W. The stock kernel treats that as M matvecs. A batched kernel loads each packed word once and uses it M times. Where does the stock kernel stall, and how much does batching recover?

**Given.** W [4096, 4096] int4, 9 MB. X is [M, 4096]. Week 2 Wednesday's measured time at M=1 is in `sessions/w02-wed.md`. MTPLX reported the stock kernel stalls at M=3 to 6 and that a ten-line fix gave 2.24x.

**Work out before running.**
1. If the stock kernel is M independent matvecs, its time at M=4 and M=8 as multiples of M=1.
2. For the batched kernel, bytes moved at M=1, 4, 8 (the weights are read once; only X and Y grow). Floor time for each.
3. Registers: the batched kernel holds M accumulators and M pre-scaled copies of x per thread. Using Tuesday's cliff, the largest M that still fits.
4. Three predicted times for the batched kernel and three for `mx.quantized_matmul`, in a 2 by 3 table.
5. At which M the batched kernel stops being bandwidth-bound, if any.

**The Ask must name:** the shapes for all three M, weights loaded once per thread and reused across M, the baseline at each M, parity at each M, and a 2 by 3 results table.

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
