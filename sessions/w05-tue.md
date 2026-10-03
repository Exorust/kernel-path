# w05-tue · Fill the arc 1 table

## The question

Before looking at any profile: what percent of its roof does each arc 1 kernel reach on each platform, and will Nsight Compute run at all inside Modal?

**Given.** Mac roof about 120 GB/s (`machine.json`, re-probe before the session). H100 roof 3.35 TB/s. Nsight Compute reads hardware counters and fails with ERR_NVGPUCTRPERM when the container lacks permission; Modal documents nothing about it.

**Write down before running.**
1. Percent of roof for: bf16 GEMV (Mac), int4 GEMV at M=1, 4, 8 (Mac), int4 GEMV at M=1, 8 (H100). Six numbers.
2. Yes or no: `ncu` works on Modal. If yes, which section you will read first and what you expect it to show for the M=1 kernel (which pipe busy, which stall reason).
3. Which of the six rows you trust least, and why.

**Run rules.** Mac: fans pinned, other apps closed, two consecutive runs within 2 percent or the number does not go in the table. H100: `ncu --set full --export w04 --page details`; if it fails, paste the verbatim error and fall back to `torch.cuda.Event` timing.

**The Ask must name:** the six rows, the run rules, the ncu command, and the fallback.

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
