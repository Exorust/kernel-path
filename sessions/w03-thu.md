# w03-thu · Coalesced tiled load (hand-written)

## The spec

**`kernels/w03/thu_tiled_load.metal`**, 30 minutes, as a `metal_kernel` body.

Input: `a`, bf16, a [256, 256] matrix. Output: `out`, bf16 [256, 256], a copy, but made through threadgroup memory: each threadgroup owns one [64, 64] tile, loads it into a `threadgroup half tile[64*64]` array, waits at a barrier, then writes it out.

Threadgroup: 1024 threads (32 simdgroups). 4096 elements per tile, so each thread loads 4 consecutive values as one `half4`. Adjacent threads must read adjacent addresses: thread t reads elements 4t .. 4t+3 of the tile in row-major order. That is what makes the load coalesced.

You need: the threadgroup's position in the grid to find the tile origin, the thread's index in the threadgroup, a `threadgroup_barrier(mem_flags::mem_threadgroup)` between load and store, and the grid and threadgroup sizes in a comment.

Check: 4096 halves = 8 KB of threadgroup memory. The limit is 32 KB.

Reference pages allowed open: https://metalworking.vercel.app/techniques/cooperative-load/ and https://metalworking.vercel.app/kernels/steel-blockloader/

## Rules
No AI. Claude does not write, fix, or suggest code today.

**Weeks 1 to 3 ramp.** Before starting, read Wednesday's AI-written kernel in `kernels/` for 5 minutes, then close it. While you write, `docs/metal-cheatsheet.md` is open, plus any reference page named above. From week 4, no cheat sheet.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
