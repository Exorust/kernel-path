# w04-tue · Vectorized loads and cp.async

## The question

On the H100, how much faster is a copy that loads 16 bytes per instruction than one that loads 4, and what does the PTX look like in each case?

**Given.** H100 memory bandwidth: 3.35 TB/s. Array: 2^28 fp32 = 1 GB, so a copy moves 2 GB and the floor is about 0.6 ms. `ld.global.v4.f32` loads four floats in one instruction. `cp.async` copies global to shared memory without passing through registers, sizes 4, 8, or 16 bytes.

**Work out before running.**
1. Whether a 4-byte-per-load kernel can reach the roof at all, and why or why not (think: load instructions issued per second, not bytes).
2. The predicted GB/s for Triton BLOCK_SIZE in {64, 1024, 16384}.
3. What you expect to find in the PTX for each: scalar `ld.global.f32`, `ld.global.v4.f32`, or something else.
4. Whether cp.async helps a plain copy. If not, name the kind of kernel where it does.

**The Ask must name:** the array size, the three block sizes, dumping `compiled_kernel.asm["ptx"]` and grepping for `ld.global`, `torch.clone` as baseline, timing with `torch.cuda.Event`, and the Modal H100 function.

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
