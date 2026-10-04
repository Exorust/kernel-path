# w09-thu · One mbarrier, by hand (scaffold and walkthrough)

## The spec

**`kernels/w09/thu_mbarrier.cu`**, 30 minutes, CUDA C++ for sm_90.

A kernel with one block. Declare a `__shared__ uint64_t bar`. Thread 0 initializes it with inline PTX `mbarrier.init.shared.b64` for an arrival count of 1. All threads sync. Thread 0 then arrives on it with `mbarrier.arrive.shared.b64`, keeping the returned state token. Every thread then loops on `mbarrier.try_wait.shared.b64` with that token until it reports done, and writes 1 to out[threadIdx.x].

You need: the address of `bar` as a 32-bit shared-space operand, a `"=l"` output for the token, a predicate result from try_wait moved into an integer, and `volatile` plus a memory clobber on each asm block.

`main` checks every out element is 1.

Reference allowed open: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html

## Rules
Scaffold and walkthrough. Claude writes the scaffold first: the Python call or `main`, the test data, the correctness check, and the kernel body with its structure written and the key expressions left as numbered `____` blanks. Then Claude walks through the blanks one at a time: what each must do, why, and where to look. I fill every blank myself. Claude gives the content of a blank only after I have made an attempt at it. `docs/metal-cheatsheet.md`, Wednesday's kernel, and the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The step I needed the most help on:
