# w09-thu · One mbarrier, by hand (hand-written)

## The spec

**`kernels/w09/thu_mbarrier.cu`**, 30 minutes, CUDA C++ for sm_90.

A kernel with one block. Declare a `__shared__ uint64_t bar`. Thread 0 initializes it with inline PTX `mbarrier.init.shared.b64` for an arrival count of 1. All threads sync. Thread 0 then arrives on it with `mbarrier.arrive.shared.b64`, keeping the returned state token. Every thread then loops on `mbarrier.try_wait.shared.b64` with that token until it reports done, and writes 1 to out[threadIdx.x].

You need: the address of `bar` as a 32-bit shared-space operand, a `"=l"` output for the token, a predicate result from try_wait moved into an integer, and `volatile` plus a memory clobber on each asm block.

`main` checks every out element is 1.

Reference allowed open: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html

## Rules
No AI. Claude does not write, fix, or suggest code today. Only the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
