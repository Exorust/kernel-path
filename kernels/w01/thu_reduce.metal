// Week 1 Thu, file 2: out[i] = sum of a[32i .. 32i+31]. a is fp32 with 2^20 elements, out has 2^15.
// One simdgroup (32 threads) per output element.
// Spec: sessions/w01-thu.md. Run: .venv/bin/python kernels/w01/thu_run.py reduce
//
// TODO 0: replace each ? below. One simdgroup per output element: cheat sheet section 8.
// grid=(?, ?, ?) threadgroup=(?, ?, ?)

// TODO 1: this thread's lane (0 to 31) and which output element its simdgroup owns.
//         Compare the first two lines of the kernel body in wed_gemv.py.

// TODO 2: load the one value of a that belongs to this lane.

// TODO 3: add the 32 lanes' values together. Cheat sheet section 5.

// TODO 4: all 32 lanes now hold the same sum. Let exactly one of them store it.
