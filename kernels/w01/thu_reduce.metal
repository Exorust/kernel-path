// Week 1 Thu, file 2: out[g] = sum of a[32g .. 32g+31]. a is fp32 with 2^20 elements, out has 2^15.
// One simdgroup (32 threads) per output element.
// Spec: sessions/w01-thu.md. Run: .venv/bin/python kernels/w01/thu_run.py reduce
// Replace every ____ . The line below is read by thu_run.py, keep its format.
//
// Blank 0: one simdgroup per output element. Cheat sheet section 8.
// grid=(____, ____, 1) threadgroup=(____, 1, 1)

uint lane = ____;           // Blank 1: 0 to 31, this thread's place in its simdgroup
uint g    = ____;           // Blank 2: which output element this simdgroup owns

float v = a[____];          // Blank 3: the one input value that belongs to this lane of group g

float total = ____;         // Blank 4: add v across the 32 lanes. Cheat sheet section 5.

if (____) out[g] = total;   // Blank 5: all 32 lanes hold the same total. Let exactly one store it.
