// =============================================================================
// aegis_mma_tile.cuh — vendored tile<>/ldmatrix/mma primitives.
// =============================================================================
//
// Focused subset of llama.cpp's ggml-cuda/mma.cuh (MIT) ported into the
// aegisllm.rs NVRTC translation unit. Provides the m16n8k16 f16.f16.f16.f16
// and f32.f16.f16.f32 Ampere/Turing tensor-core building blocks that the
// register-softmax FlashAttention prefill kernel
// (`attention_prefill_regsmx_hdim512.cu`) will consume.
//
// SCOPE (intentionally tiny — D=512 wide-config-only): only the I_MAJOR layout,
// half2-element tiles for Q/K/P operands, float-element tile for the KQ_C
// accumulator, and the f32/f16 m16n8k16 MMAs. We skip Volta, bf16, AMD WMMA/
// MFMA, integer (s8/s32), TF32, FP4 block-scaled, J_MAJOR, MIRRORED, and the
// `load_generic` fallback path entirely.
//
// PORTABILITY: kept the Turing m16n8k8 decomposition (__CUDA_ARCH__ < 800)
// inline so the source still reads as "from llama.cpp" even though our
// target (SM120 / cc12.0) always takes the Ampere+ path. The whole file body
// is guarded behind `#if __CUDA_ARCH__ >= 800` to match the surrounding
// kernels in this directory.
//
// CRITICAL CORRECTNESS NOTES (do NOT "fix" these):
//   * ldmatrix.x4.trans binds output regs {r0, r2, r1, r3} — the {r2,r1} swap
//     is intentional and a wire of the PTX semantics, NOT a bug.
//   * The m16n8k16 .f32.f16.f16.f32 instruction takes D=4 f32 regs (in/out),
//     A=4 half2 regs, B=2 half2 regs. Preserve the {%0..%3}/{%4..%7}/{%8,%9}
//     binding pattern exactly.
//   * `int *xi = (int *) t.x` reinterprets the half2[ne] storage as int[ne]
//     for ldmatrix register binding — this is deliberate, half2==32 bits.
// =============================================================================

#ifndef AEGIS_MMA_TILE_CUH
#define AEGIS_MMA_TILE_CUH

#if __CUDA_ARCH__ >= 800

namespace aegis_mma {

    // Subset of llama.cpp's data_layout enum. We only need I_MAJOR.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        DATA_LAYOUT_I_MAJOR = 0,
    };

    template <int I_, int J_, typename T, data_layout ds_ = DATA_LAYOUT_I_MAJOR>
    struct tile {};

    // -------------------------------------------------------------------------
    // tile<I, J, float, I_MAJOR> — KQ_C accumulator family.
    // -------------------------------------------------------------------------
    template <int I_, int J_>
    struct tile<I_, J_, float, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        static constexpr int ne = I * J / 32;
        float x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, float>::get_i(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                __trap();
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, float>::get_j(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                __trap();
                return -1;
            }
        }
    };

    // -------------------------------------------------------------------------
    // tile<I, J, half2, I_MAJOR> — Q/K/P/V operand family.
    // -------------------------------------------------------------------------
    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        static constexpr int ne = I * J / 32u;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l / 4) * 16) + ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                __trap();
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l & 2) * 2) + (threadIdx.x % 4);
            } else {
                __trap();
                return -1;
            }
        }
    };

    // =========================================================================
    // ldmatrix loaders (cc>=7.5 ldmatrix.sync.aligned).
    // All pointers MUST be shared-memory and 16-byte aligned (PTX requirement).
    // =========================================================================

    // tile<8, 8, T> — ldmatrix.x2 (2 regs/lane).
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#if (__CUDA_ARCH__ >= 750)
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        __trap();
#endif
    }

    // tile<16, 4, T> — ldmatrix.x2 with a different address formula.
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
#if (__CUDA_ARCH__ >= 750)
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        __trap();
#endif
    }

    // tile<16, 8, T, dl> — ldmatrix.x4 (4 regs/lane).
    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if (__CUDA_ARCH__ >= 750)
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
#else
        __trap();
#endif
    }

    // tile<16, 8, T> — ldmatrix.x4.trans (transposed in-place by the hardware).
    // NOTE: the output register binding is {r0, r2, r1, r3} — the swap of
    // r1<->r2 is intentional and reflects how the .trans variant lays out the
    // result registers relative to the matrix lane mapping. DO NOT "normalize"
    // this back to {r0,r1,r2,r3}.
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix_trans(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#if (__CUDA_ARCH__ >= 750)
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "l"(xs));
#else
        __trap();
#endif
    }

    // =========================================================================
    // Tensor-core MMAs (Ampere+ primary; Turing m8n8k8 decomposition retained).
    // =========================================================================

    // D(f16) = A(f16) * B(f16) + C(f16) — m16n8k16, single issue (n=8 panel).
    // D: tile<16, 4, half2> ne=2 -> 2 int regs/lane.
    // A: tile<16, 8, half2> ne=4 -> 4 int regs/lane.
    // B: tile<8, 8, half2>  ne=2 -> 2 int regs/lane.
    static __device__ __forceinline__ void mma(
            tile<16, 4, half2> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
#if (__CUDA_ARCH__ >= 750)
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= 800
#else
        __trap();
#endif // (__CUDA_ARCH__ >= 750)
    }

    // D(f16) = A(f16) * B(f16) + C(f16) — two m16n8k16 issues to fill ne=4 D.
    // D: tile<16, 8, half2> ne=4 -> 4 int regs/lane.
    // A: tile<16, 8, half2> ne=4 -> 4 int regs/lane.
    // B: tile<16, 8, half2> ne=4 -> 4 int regs/lane (used as 2 columns of a k=16 panel,
    //                                                so the two issues consume {B0,B2} then {B1,B3}).
    static __device__ __forceinline__ void mma(
            tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#if (__CUDA_ARCH__ >= 750)
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
        // On Turing m16n8k16 mma is not available, use 4x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __CUDA_ARCH__ >= 800
#else
        __trap();
#endif // (__CUDA_ARCH__ >= 750)
    }

    // D(f32) = A(f16) * B(f16) + C(f32) — m16n8k16.
    // D: tile<16, 8, float> ne=4 float regs/lane.
    // A: tile<16, 8, half2> ne=4 -> 4 int regs/lane.
    // B: tile<8, 8, half2>  ne=2 -> 2 int regs/lane.
    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
#if (__CUDA_ARCH__ >= 750)
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= 800
#else
        __trap();
#endif // (__CUDA_ARCH__ >= 750)
    }

} // namespace aegis_mma

#endif // __CUDA_ARCH__ >= 800

#endif // AEGIS_MMA_TILE_CUH
