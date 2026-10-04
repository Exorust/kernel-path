// Week 1 Thu, file 1: out = a + b. fp32, 2^20 elements, one element per thread.
// Spec: sessions/w01-thu.md. Run: .venv/bin/python kernels/w01/thu_run.py vadd
// Replace every ____ . The line below is read by thu_run.py, keep its format.
//
// Blank 0: grid = how many threads in total. threadgroup = how they are bundled. Cheat sheet section 4.
// grid=(____, 1, 1) threadgroup=(____, 1, 1)

uint i = ____;              // Blank 1: which element this thread owns. Cheat sheet section 2.

float x = ____;             // Blank 2: load this thread's element of a
float y = ____;             // Blank 3: load this thread's element of b

out[____] = ____;           // Blank 4: where the result goes, and what it is
