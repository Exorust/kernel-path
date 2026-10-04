// Week 1 Thu, file 1: out = a + b. fp32, 2^20 elements, one element per thread.
// Spec: sessions/w01-thu.md. Run: .venv/bin/python kernels/w01/thu_run.py vadd
//
// TODO 0: replace each ? below. grid = how many threads in total, on each axis.
//         threadgroup = how they are bundled. Cheat sheet section 4.
// grid=(?, ?, ?) threadgroup=(?, ?, ?)

// TODO 1: which element is this thread responsible for? Cheat sheet section 2.

// TODO 2: load that element of a and of b, add them, store into out. Two loads, one store.
