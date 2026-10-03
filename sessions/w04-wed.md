# w04-wed · Triton int4 GEMV vs Marlin

## The question

Port week 2's fused int4 GEMV to Triton and compare it with Marlin. How large is the gap at M=1 and at M=8, and what in Marlin's layout explains it?

**Given.** W [4096, 4096], 4 bits, group_size 64, MLX-style packing (8 nibbles per uint32). H100 roof 3.35 TB/s; 9 MB of weights is about 2.7 microseconds at the roof, so launch overhead and latency matter more than on the Mac. Marlin's README argues a naive int4 kernel keeps its advantage only to batch 1 or 2, and that its striped layout and double buffering extend that 4 to 8 times.

**Work out before running.**
1. At 2.7 microseconds of pure bandwidth time, what else bounds the kernel? Name two candidates.
2. Why int4 at M=8 wants tensor cores, and why a matvec-shaped kernel cannot use them.
3. Predicted time for your Triton kernel and for Marlin at M=1 and M=8, in a 2 by 2 table.
4. The one layout decision in Marlin you expect to matter most.

Note: Marlin uses its own weight format. The comparison is same shape and bit width, not same bytes. Say so in the After section.

**The Ask must name:** the shape and format, the two-term identity from week 2, Marlin as baseline through its Python layer, M=1 and M=8, PTX dump with the dequant located, and the Modal H100 function.

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
