# w10-tue · Fill the arc 2 table

## The question

Collect the arc 2 numbers into the README table, then take one look at Blackwell. No kernel is written today.

**Fill from earlier sessions.**
1. Hand scan vs mlx-lm scan, T=4096 (week 6 Thursday).
2. Single chunk vs fp64 reference, NMSE (week 7).
3. Full chunkwise vs scan at T=4096 and T=8192, with NMSE and CV (week 8). State plainly whether the 2x gate was met.
4. Directed Triton vs FLA at T=4096 (week 9).

**Then answer from the Colfax Tensor Memory post and PTX ISA §9.7.18.1.**
5. What replaced wgmma on datacenter Blackwell, and how many threads launch it.
6. The Tensor Memory geometry per CTA, in columns, lanes, and bits.
7. Why a kernel written for an RTX Pro Blackwell card tells you nothing about a B200.

The "Before" section today is one line: which of rows 1 to 4 you expect a reader to challenge first.

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
