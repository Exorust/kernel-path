# Metal for kernel bodies: the cheat sheet

Everything on this page was run as a real kernel on MLX 0.32.2 on 2026-10-03. Open on every Thursday.

A kernel body in `mx.fast.metal_kernel` is C. You write only the inside of the function. MLX writes the signature. About a dozen words are specific to Metal, and they are all here.

## 1. What you are given

| You get | What it is |
|---|---|
| each name in `input_names` | a read-only pointer to that array, already the right type: `a[i]` reads element i |
| each name in `output_names` | a writable pointer: `out[i] = v;` |
| each name in `template=[("T", mx.float32), ("KK", 3)]` | a type `T` or a compile-time integer `KK` |

All arrays are flat. A [rows, cols] matrix is indexed `W[row * cols + col]`.

## 2. Where am I?

Just use these names in the body. MLX adds them to the signature when it sees them.

| Name | Type | Meaning |
|---|---|---|
| `thread_position_in_grid` | `uint3` | this thread's coordinates in the whole launch: `.x`, `.y`, `.z` |
| `thread_index_in_simdgroup` | `uint` | 0 to 31: which lane of its simdgroup this thread is |
| `thread_index_in_threadgroup` | `uint` | 0 to (threadgroup size − 1) |
| `simdgroup_index_in_threadgroup` | `uint` | which simdgroup inside the threadgroup |
| `threadgroup_position_in_grid` | `uint3` | which threadgroup this is |

## 3. Types

`float` (32-bit), `half` (16-bit), `int`, `uint`, `ushort`. Vectors: `float4`, `half4`, with fields `.x .y .z .w`.
Casts look like function calls: `float(h)`, `half(f)`, `float(KK)`. Float literals end in `f`: `2.0f`.

## 4. The smallest kernel

Doubles every element. One thread per element.

```metal
uint i = thread_position_in_grid.x;
out[i] = 2.0f * a[i];
```

Python side: `grid=(n, 1, 1), threadgroup=(32, 1, 1)`. `grid` is how many threads in total. `threadgroup` is how they are bundled.

## 5. The 32 lanes working together

These act across the 32 threads of one simdgroup. No memory is touched.

| Call | Result, in every lane |
|---|---|
| `simd_sum(v)` | the sum of `v` over all 32 lanes |
| `simd_max(v)`, `simd_min(v)` | the max or min over all 32 lanes |
| `simd_shuffle_xor(v, (ushort)m)` | the value of `v` held by lane `(my lane) XOR m` |

After a reduction all 32 lanes hold the same answer, so let one lane write it:

```metal
float total = simd_sum(v);
if (thread_index_in_simdgroup == 0) out[g] = total;
```

## 6. Loops and loads

A loop is a C loop. To read four values in one load, cast the pointer:

```metal
float4 v = *((const device float4*)(a + 4 * i));
float s = v.x + v.y + v.z + v.w;
```

`device` means "this pointer is into GPU buffer memory". You only type it in a cast like this one.

## 7. Scratch memory shared by one threadgroup

Declare it, write it, wait for everyone, then read what others wrote. At most 32 KB.

```metal
threadgroup float buf[64];
uint t = thread_index_in_threadgroup;
buf[t] = a[thread_position_in_grid.x];
threadgroup_barrier(mem_flags::mem_threadgroup);   // every thread has finished writing
out[thread_position_in_grid.x] = buf[63 - t];
```

## 8. Two-dimensional launches

`grid=(32, rows, 1), threadgroup=(32, 1, 1)` gives one simdgroup per row: `.x` is the lane (0 to 31) and `.y` is the row.

## 9. CUDA to Metal

| CUDA | Metal |
|---|---|
| `threadIdx.x`, `blockIdx.x` | `thread_position_in_grid` and the names in section 2 |
| warp | simdgroup |
| block | threadgroup |
| `__shared__ float buf[64];` | `threadgroup float buf[64];` |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__shfl_xor_sync(mask, v, m)` | `simd_shuffle_xor(v, (ushort)m)` |
| warp reduction by hand | `simd_sum(v)` |

## 10. When it does not compile

Pass `verbose=True` to the kernel call to print the full generated function. The compiler error names a line in that printout, not in your body.
