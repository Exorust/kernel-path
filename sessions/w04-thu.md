# w04-thu · One inline PTX vector load (scaffold and walkthrough)

## The spec

**`kernels/w04/thu_ptx_load.cu`**, 30 minutes, CUDA C++.

A kernel `copy4(const float* in, float* out, int n)` where each thread loads four consecutive floats with a single inline `ld.global.v4.f32` and stores them with ordinary C. Thread i handles elements 4i .. 4i+3.

You need: the `asm volatile("..." : outputs : inputs);` form, four `"=f"` output operands for the floats, one `"l"` input operand for the 64-bit address, the operand placeholders `%0` to `%4`, and braces around the four destination registers in the PTX vector syntax.

Plus a `main` that fills 2^20 floats, runs the kernel, and checks out == in.

Reference allowed open: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html (§1.1 for constraints, §1.2 Pitfalls for `volatile` and the memory clobber).

## Rules
Scaffold and walkthrough. Claude writes the scaffold first: the Python call or `main`, the test data, the correctness check, and the kernel body as numbered `TODO` steps with no kernel code in them. Then Claude walks through the steps one at a time: what the step must do, why, and where to look. I type every kernel line myself. Claude gives a line itself only after I have made an attempt at it. `docs/metal-cheatsheet.md`, Wednesday's kernel, and the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The step I needed the most help on:
