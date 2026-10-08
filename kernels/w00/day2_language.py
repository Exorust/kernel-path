"""Week 0, day 2: MSL, the language a kernel body is written in.

    .venv/bin/python kernels/w00/day2_language.py

Six short parts, two finished kernels each: a FIRST that shows one feature, and a SECOND that uses it
again in a different way. Read the code, run the file, and match each printed row to the line that made it.
Parts E and F are what weeks 2 and 3 will need.
Reading: sessions/w00-d2.md. Cheat sheet: docs/metal-cheatsheet.md, sections 3 and 6.
"""
import mlx.core as mx
from helpers import run_parts, show

N = 8
a = mx.array([-3, -2, -1, 0, 1, 2, 3, 4], dtype=mx.float32)
squares = mx.array([0, 1, 4, 9, 16, 25, 36, 49], dtype=mx.float32)
c = mx.arange(8).astype(mx.float32)                                 # 0, 1, 2, ... 7
word = mx.array([0x76543210], dtype=mx.uint32)                      # one 32-bit word that holds eight 4-bit numbers


def launch(name, source, inputs, names, threads=N, out_size=N):
    kernel = mx.fast.metal_kernel(name=name, input_names=names, output_names=["out"], source=source)
    return kernel(inputs=inputs, grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(out_size,)], output_dtypes=[mx.float32])[0]


def same(out, want):
    return "both rows match" if mx.allclose(out, want, atol=1e-5).item() else "second does not match want"


# ------------------------------------------------------------------ Part A: numbers and casts
FIRST_A = """
    uint i = thread_position_in_grid.x;     // uint: a whole number that is never negative
    float x = float(i);                     // float(...) turns a whole number into a float. This is a cast.
    out[i] = x * 0.5f;                      // 0.5f is a float literal. Metal has no double type.
"""
SECOND_A = """
    uint i = thread_position_in_grid.x;
    out[i] = float(i * i);                  // i times i is still a whole number; the cast comes last
"""


def part_a():
    show("first   ", launch("a1", FIRST_A, [a], ["a"]))
    out = launch("a2", SECOND_A, [a], ["a"])
    show("second  ", out)
    return same(out, c * c)


# ------------------------------------------------------------------ Part B: if
FIRST_B = """
    uint i = thread_position_in_grid.x;
    float x = a[i];
    if (x < 0.0f) {                         // exactly as in C
        x = 0.0f;                           // negative values become zero
    }
    out[i] = x;
"""
SECOND_B = """
    uint i = thread_position_in_grid.x;
    float x = a[i];
    if (x > 2.0f) {
        x = 2.0f;                           // values above 2 become 2
    }
    out[i] = x;
"""


def part_b():
    show("a       ", a)
    show("first   ", launch("b1", FIRST_B, [a], ["a"]))
    out = launch("b2", SECOND_B, [a], ["a"])
    show("second  ", out)
    return same(out, mx.minimum(a, 2.0))


# ------------------------------------------------------------------ Part C: built-in math
FIRST_C = """
    uint i = thread_position_in_grid.x;
    out[i] = sqrt(squares[i]);              // Metal has its own math functions: sqrt, exp, abs, min, max, ...
"""
SECOND_C = """
    uint i = thread_position_in_grid.x;
    float x = a[i];
    out[i] = 1.0f / (1.0f + exp(-x));       // the sigmoid function, written with exp
"""


def part_c():
    show("squares ", squares)
    show("first   ", launch("c1", FIRST_C, [squares], ["squares"]))
    out = launch("c2", SECOND_C, [a], ["a"])
    print("  second   " + "  ".join(f"{v:.3f}" for v in out.tolist()))
    print("  (a)      " + "  ".join(f"{v:>5.0f}" for v in a.tolist()))
    return same(out, mx.sigmoid(a))


# ------------------------------------------------------------------ Part D: loops
FIRST_D = """
    uint i = thread_position_in_grid.x;
    float x = a[i];
    float p = 1.0f;
    for (uint k = 0; k < 3; ++k) {          // a C for loop: three turns
        p = p * x;                          // p ends as x times x times x
    }
    out[i] = p;
"""
# Here there are only 2 threads and out has 2 elements. Thread i adds up 4 neighbours: c[4i] to c[4i + 3].
SECOND_D = """
    uint i = thread_position_in_grid.x;
    float s = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        s += c[4 * i + j];                  // element j of my group of four; my group starts at 4 * i
    }
    out[i] = s;
"""


def part_d():
    show("a       ", a)
    show("first   ", launch("d1", FIRST_D, [a], ["a"]))
    show("c       ", c)
    out = launch("d2", SECOND_D, [c], ["c"], threads=2, out_size=2)
    show("second  ", out)
    return same(out, c.reshape(2, 4).sum(axis=1))


# ------------------------------------------------------------------ Part E: whole numbers and bits (week 2 needs this)
FIRST_E = """
    uint i   = thread_position_in_grid.x;
    uint row = i / 4;                       // dividing whole numbers drops the remainder: 7 / 4 is 1
    uint col = i % 4;                       // % is the remainder: 7 % 4 is 3
    out[i] = float(10 * row + col);         // write "row, column" as a two-digit number
"""
# word[0] is 0x76543210. Each hex digit is 4 bits, so the word holds eight small numbers: 0 in the lowest 4 bits, then 1, ...
SECOND_E = """
    uint i = thread_position_in_grid.x;
    uint q = (word[0] >> (4 * i)) & 0xF;    // shift my 4 bits down to the bottom, then keep only those 4 bits
    out[i] = float(q);
"""


def part_e():
    show("first   ", launch("e1", FIRST_E, [a], ["a"]))
    out = launch("e2", SECOND_E, [word], ["word"])
    show("second  ", out)
    return same(out, c)


# ------------------------------------------------------------------ Part F: four values at once (week 3 needs this)
# float4 is four floats held together. 2 threads here. Thread i owns c[4i] to c[4i + 3].
FIRST_F = """
    uint i = thread_position_in_grid.x;
    float4 v = *((const device float4*)(c + 4 * i));        // read 4 floats with one load
    out[i] = v.x + v.y + v.z + v.w;                         // the four parts are called x, y, z and w
"""
SECOND_F = """
    uint i = thread_position_in_grid.x;
    float4 v = *((const device float4*)(c + 4 * i));
    float4 d = v * 2.0f;                                    // arithmetic on a float4 acts on all four parts
    *((device float4*)(out + 4 * i)) = d;                   // write 4 floats with one store
"""


def part_f():
    show("c       ", c)
    show("first   ", launch("f1", FIRST_F, [c], ["c"], threads=2, out_size=2))
    out = launch("f2", SECOND_F, [c], ["c"], threads=2, out_size=8)
    show("second  ", out)
    return same(out, 2 * c)


if __name__ == "__main__":
    run_parts([("A  numbers and casts", part_a),
               ("B  if", part_b),
               ("C  built-in math", part_c),
               ("D  loops", part_d),
               ("E  whole numbers and bits (for week 2)", part_e),
               ("F  four values at once (for week 3)", part_f)])
