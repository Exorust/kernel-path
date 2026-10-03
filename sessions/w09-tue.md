# w09-tue · What bounds FLA's kernel on an H100?

## The question

Run the reference Triton implementation at your shapes under the profiler. Which hardware pipe is the limit, and what are the warps waiting on?

**Given.** FLA `chunk_gated_delta_rule`, shape B=1, Hk=16, Hv=32, Dk=Dv=128, T=4096, C=64. Its stages: kkt, solve_tril, recompute_w_u, then the chunk scan. Nsight sections to read: ComputeWorkloadAnalysis (pipe utilization), SchedulerStats (issue slots), MemoryWorkloadAnalysis.

**Write down before running.**
1. Which of the four stages you expect to take the most time on an H100, and whether that matches the Mac.
2. For that stage: tensor pipe, FP pipe, or memory?
3. The top warp stall reason you expect.
4. Whether B=1 saturates an H100 at all. Count the blocks launched against 132 SMs.

**If `ncu` is not permitted on Modal:** time each stage separately with `torch.cuda.Event` and answer item 1 only.

**The Ask must name:** the FLA call and shape, `ncu --set full --export fla --page details`, per-kernel rows in the report, and the fallback.

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
