# w03-thu · Coalesced tiled load (scaffold and walkthrough)

## The spec

**`kernels/w03/thu_tiled_load.py`**, 30 minutes, as a `metal_kernel` body.

Input: `a`, bf16, a [256, 256] matrix. Output: `out`, bf16 [256, 256], a copy, but made through threadgroup memory: each threadgroup owns one [64, 64] tile, loads it into a `threadgroup half tile[64*64]` array, waits at a barrier, then writes it out.

Threadgroup: 1024 threads (32 simdgroups). 4096 elements per tile, so each thread loads 4 consecutive values as one `half4`. Adjacent threads must read adjacent addresses: thread t reads elements 4t .. 4t+3 of the tile in row-major order. That is what makes the load coalesced.

You need: the threadgroup's position in the grid to find the tile origin, the thread's index in the threadgroup, a `threadgroup_barrier(mem_flags::mem_threadgroup)` between load and store, and the grid and threadgroup sizes in a comment.

Check: 4096 halves = 8 KB of threadgroup memory. The limit is 32 KB.

Reference pages allowed open: https://metalworking.vercel.app/techniques/cooperative-load/ and https://metalworking.vercel.app/kernels/steel-blockloader/

## Rules
Scaffold and walkthrough. Claude writes the scaffold first: the Python call or `main`, the test data, the correctness check, and the kernel body with its structure written and the key expressions left as numbered `____` blanks. Then Claude walks through the blanks one at a time: what each must do, why, and where to look. I fill every blank myself. Claude gives the content of a blank only after I have made an attempt at it. `docs/metal-cheatsheet.md`, Wednesday's kernel, and the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The step I needed the most help on:
