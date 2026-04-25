#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace aegis_cutlass_bridge {

struct Sm120Fp4ConfigM256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShapeMNK = Shape<_128, _128, _128>;
};

struct Sm120Fp4ConfigDefault {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShapeMNK = Shape<_256, _128, _128>;
};

template <typename Config>
struct Fp4GemmSm120F32 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = float;
  using ElementC = float;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, typename Config::PerSmTileShapeMNK,
          typename Config::ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator,
          typename Config::MmaTileShape, typename Config::ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments make_arguments(const void* a, const void* b,
                                        const void* a_sf, const void* b_sf,
                                        float* d, int m, int n, int k,
                                        float alpha) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto layout_sfa =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<ElementA const*>(a), stride_a,
       static_cast<ElementB const*>(b), stride_b,
       static_cast<ElementSFA const*>(a_sf), layout_sfa,
       static_cast<ElementSFB const*>(b_sf), layout_sfb},
      {{alpha, 0.0f}, d, stride_d, d, stride_d}};
}

void write_error(char* error, size_t error_len, const char* message) {
  if (error == nullptr || error_len == 0) {
    return;
  }
  std::snprintf(error, error_len, "%s", message);
}

int validate_problem(int m, int n, int k, char* error, size_t error_len) {
  if (m <= 0 || n <= 0 || k <= 0) {
    write_error(error, error_len, "m, n and k must be positive");
    return 1;
  }
  if ((k % 32) != 0) {
    write_error(error, error_len, "CUTLASS FP4 GEMM requires K % 32 == 0");
    return 2;
  }
  if ((n % 32) != 0) {
    write_error(error, error_len, "CUTLASS FP4 GEMM requires N % 32 == 0");
    return 3;
  }
  return 0;
}

template <typename Gemm>
int workspace_size_for(const void* a, const void* b, const void* a_sf,
                       const void* b_sf, float* d, int m, int n, int k,
                       size_t* workspace_bytes, char* error,
                       size_t error_len) {
  auto args = make_arguments<Gemm>(a, b, a_sf, b_sf, d, m, n, k, 1.0f);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 10 + static_cast<int>(status);
  }
  *workspace_bytes = Gemm::get_workspace_size(args);
  return 0;
}

template <typename Gemm>
int run_gemm(const void* a, const void* b, const void* a_sf, const void* b_sf,
             float* d, void* workspace, size_t workspace_bytes, int m, int n,
             int k, float alpha, cudaStream_t stream, char* error,
             size_t error_len) {
  auto args = make_arguments<Gemm>(a, b, a_sf, b_sf, d, m, n, k, alpha);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 20 + static_cast<int>(status);
  }
  size_t required_workspace = Gemm::get_workspace_size(args);
  if (workspace_bytes < required_workspace) {
    write_error(error, error_len, "CUTLASS FP4 GEMM workspace is too small");
    return 30;
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 40 + static_cast<int>(status);
  }
  status = gemm.run(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 50 + static_cast<int>(status);
  }
  return 0;
}

using GemmM256 = typename Fp4GemmSm120F32<Sm120Fp4ConfigM256>::Gemm;
using GemmDefault = typename Fp4GemmSm120F32<Sm120Fp4ConfigDefault>::Gemm;

int select_workspace_size(const void* a, const void* b, const void* a_sf,
                          const void* b_sf, float* d, int m, int n, int k,
                          size_t* workspace_bytes, char* error,
                          size_t error_len) {
  uint32_t mp2 = std::max<uint32_t>(16, static_cast<uint32_t>(m - 1));
  mp2 |= mp2 >> 1;
  mp2 |= mp2 >> 2;
  mp2 |= mp2 >> 4;
  mp2 |= mp2 >> 8;
  mp2 |= mp2 >> 16;
  mp2 += 1;
  if (mp2 <= 256) {
    return workspace_size_for<GemmM256>(a, b, a_sf, b_sf, d, m, n, k,
                                        workspace_bytes, error, error_len);
  }
  return workspace_size_for<GemmDefault>(a, b, a_sf, b_sf, d, m, n, k,
                                         workspace_bytes, error, error_len);
}

int select_run_gemm(const void* a, const void* b, const void* a_sf,
                    const void* b_sf, float* d, void* workspace,
                    size_t workspace_bytes, int m, int n, int k, float alpha,
                    cudaStream_t stream, char* error, size_t error_len) {
  uint32_t mp2 = std::max<uint32_t>(16, static_cast<uint32_t>(m - 1));
  mp2 |= mp2 >> 1;
  mp2 |= mp2 >> 2;
  mp2 |= mp2 >> 4;
  mp2 |= mp2 >> 8;
  mp2 |= mp2 >> 16;
  mp2 += 1;
  if (mp2 <= 256) {
    return run_gemm<GemmM256>(a, b, a_sf, b_sf, d, workspace, workspace_bytes,
                              m, n, k, alpha, stream, error, error_len);
  }
  return run_gemm<GemmDefault>(a, b, a_sf, b_sf, d, workspace,
                               workspace_bytes, m, n, k, alpha, stream, error,
                               error_len);
}

__device__ __forceinline__ float e2m1_value(unsigned idx) {
  static constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f,
                                       4.0f, 6.0f, 0.0f, -0.5f, -1.0f, -1.5f,
                                       -2.0f, -3.0f, -4.0f, -6.0f};
  return values[idx & 0x0f];
}

__device__ __forceinline__ unsigned best_e2m1(float value, float scale) {
  if (scale == 0.0f) {
    return 0;
  }
  float scaled = value / scale;
  unsigned best = 0;
  float best_err = fabsf(scaled - e2m1_value(0));
  for (unsigned idx = 1; idx < 16; ++idx) {
    float err = fabsf(scaled - e2m1_value(idx));
    if (err < best_err) {
      best = idx;
      best_err = err;
    }
  }
  return best;
}

__device__ __forceinline__ size_t swizzled_scale_offset(int row, int k_scale_idx,
                                                        int scale_k_tiles) {
  int m_tile_idx = row >> 7;
  int outer_m_idx = row & 31;
  int inner_m_idx = (row >> 5) & 3;
  int k_tile_idx = k_scale_idx >> 2;
  int inner_k_idx = k_scale_idx & 3;
  return (static_cast<size_t>(m_tile_idx) * scale_k_tiles + k_tile_idx) * 512u +
         static_cast<size_t>(outer_m_idx) * 16u +
         static_cast<size_t>(inner_m_idx) * 4u + inner_k_idx;
}

__global__ void quantize_f32_to_e2m1_ue4m3_kernel(
    const float* __restrict__ input, int rows, int cols,
    uint8_t* __restrict__ payload, uint8_t* __restrict__ scales,
    int scale_cols, int padded_scale_cols, int scale_k_tiles) {
  int row = blockIdx.y;
  int scale_col = blockIdx.x;
  if (scale_col >= padded_scale_cols) {
    return;
  }
  if (row >= rows || scale_col >= scale_cols) {
    scales[swizzled_scale_offset(row, scale_col, scale_k_tiles)] = 0;
    return;
  }

  int col_base = scale_col * 16;
  float amax = 0.0f;
  float values[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float value = input[static_cast<size_t>(row) * cols + col_base + i];
    values[i] = value;
    amax = fmaxf(amax, fabsf(value));
  }

  float scale = (amax > 0.0f) ? (amax / 6.0f) : 0.0f;
  __nv_fp8_e4m3 encoded_scale(scale);
  uint8_t scale_byte = reinterpret_cast<uint8_t&>(encoded_scale);
  scales[swizzled_scale_offset(row, scale_col, scale_k_tiles)] = scale_byte;
  float decoded_scale = static_cast<float>(encoded_scale);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    unsigned lo = best_e2m1(values[pair * 2], decoded_scale);
    unsigned hi = best_e2m1(values[pair * 2 + 1], decoded_scale);
    payload[static_cast<size_t>(row) * (cols / 2) + scale_col * 8 + pair] =
        static_cast<uint8_t>(lo | (hi << 4));
  }
}

__global__ void swiglu_quantize_f32_to_e2m1_ue4m3_kernel(
    const float* __restrict__ gate, const float* __restrict__ up, int rows,
    int cols, uint8_t* __restrict__ payload, uint8_t* __restrict__ scales,
    int scale_cols, int padded_scale_cols, int scale_k_tiles) {
  int row = blockIdx.y;
  int scale_col = blockIdx.x;
  int lane = threadIdx.x;
  if (scale_col >= padded_scale_cols) {
    return;
  }
  if (row >= rows || scale_col >= scale_cols) {
    if (lane == 0) {
      scales[swizzled_scale_offset(row, scale_col, scale_k_tiles)] = 0;
    }
    return;
  }

  int col_base = scale_col * 16;
  __shared__ float values[16];
  __shared__ float decoded_scale_shared;
  float value = 0.0f;
  if (lane < 16) {
    size_t offset = static_cast<size_t>(row) * cols + col_base + lane;
    float x = gate[offset];
    value = (x / (1.0f + expf(-x))) * up[offset];
    values[lane] = value;
  }
  float amax = lane < 16 ? fabsf(value) : 0.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }

  if (lane == 0) {
    float scale = (amax > 0.0f) ? (amax / 6.0f) : 0.0f;
    __nv_fp8_e4m3 encoded_scale(scale);
    uint8_t scale_byte = reinterpret_cast<uint8_t&>(encoded_scale);
    scales[swizzled_scale_offset(row, scale_col, scale_k_tiles)] = scale_byte;
    decoded_scale_shared = static_cast<float>(encoded_scale);
  }
  __syncwarp();

  if (lane < 8) {
    unsigned lo = best_e2m1(values[lane * 2], decoded_scale_shared);
    unsigned hi = best_e2m1(values[lane * 2 + 1], decoded_scale_shared);
    payload[static_cast<size_t>(row) * (cols / 2) + scale_col * 8 + lane] =
        static_cast<uint8_t>(lo | (hi << 4));
  }
}

}  // namespace aegis_cutlass_bridge

using namespace aegis_cutlass_bridge;

extern "C" int aegis_cutlass_fp4_sm120_workspace_size(
    int m, int n, int k, size_t* workspace_bytes, char* error,
    size_t error_len) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  int validation = validate_problem(m, n, k, error, error_len);
  if (validation != 0) {
    return validation;
  }
  if (workspace_bytes == nullptr) {
    write_error(error, error_len, "workspace_bytes pointer is null");
    return 4;
  }
  return select_workspace_size(nullptr, nullptr, nullptr, nullptr, nullptr, m, n,
                               k, workspace_bytes, error, error_len);
#else
  write_error(error, error_len, "CUTLASS SM120 FP4 support is not compiled");
  return 1000;
#endif
}

extern "C" int aegis_cutlass_fp4_sm120_gemm_f32(
    const void* a, const void* b, const void* a_sf, const void* b_sf, float* d,
    void* workspace, size_t workspace_bytes, int m, int n, int k, float alpha,
    void* stream, char* error, size_t error_len) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  int validation = validate_problem(m, n, k, error, error_len);
  if (validation != 0) {
    return validation;
  }
  if (a == nullptr || b == nullptr || a_sf == nullptr || b_sf == nullptr ||
      d == nullptr || workspace == nullptr) {
    write_error(error, error_len, "CUTLASS FP4 GEMM received a null pointer");
    return 5;
  }
  return select_run_gemm(a, b, a_sf, b_sf, d, workspace, workspace_bytes, m, n,
                         k, alpha, reinterpret_cast<cudaStream_t>(stream),
                         error, error_len);
#else
  write_error(error, error_len, "CUTLASS SM120 FP4 support is not compiled");
  return 1000;
#endif
}

extern "C" int aegis_cutlass_fp4_quantize_f32(
    const float* input, int rows, int cols, uint8_t* payload, uint8_t* scales,
    void* stream, char* error, size_t error_len) {
  if (input == nullptr || payload == nullptr || scales == nullptr) {
    write_error(error, error_len, "CUTLASS FP4 quantize received a null pointer");
    return 1;
  }
  if (rows <= 0 || cols <= 0 || (cols % 32) != 0) {
    write_error(error, error_len, "CUTLASS FP4 quantize requires cols % 32 == 0");
    return 2;
  }
  int scale_cols = cols / 16;
  int padded_scale_cols = ((scale_cols + 3) / 4) * 4;
  int scale_k_tiles = padded_scale_cols / 4;
  int padded_rows = ((rows + 127) / 128) * 128;
  dim3 grid(padded_scale_cols, padded_rows, 1);
  dim3 block(1, 1, 1);
  quantize_f32_to_e2m1_ue4m3_kernel<<<grid, block, 0,
                                      reinterpret_cast<cudaStream_t>(stream)>>>(
      input, rows, cols, payload, scales, scale_cols, padded_scale_cols,
      scale_k_tiles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    write_error(error, error_len, cudaGetErrorString(err));
    return 3;
  }
  return 0;
}

extern "C" int aegis_cutlass_fp4_swiglu_quantize_f32(
    const float* gate, const float* up, int rows, int cols, uint8_t* payload,
    uint8_t* scales, void* stream, char* error, size_t error_len) {
  if (gate == nullptr || up == nullptr || payload == nullptr || scales == nullptr) {
    write_error(error, error_len,
                "CUTLASS FP4 SwiGLU quantize received a null pointer");
    return 1;
  }
  if (rows <= 0 || cols <= 0 || (cols % 32) != 0) {
    write_error(error, error_len,
                "CUTLASS FP4 SwiGLU quantize requires cols % 32 == 0");
    return 2;
  }
  int scale_cols = cols / 16;
  int padded_scale_cols = ((scale_cols + 3) / 4) * 4;
  int scale_k_tiles = padded_scale_cols / 4;
  int padded_rows = ((rows + 127) / 128) * 128;
  dim3 grid(padded_scale_cols, padded_rows, 1);
  dim3 block(32, 1, 1);
  swiglu_quantize_f32_to_e2m1_ue4m3_kernel<<<
      grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      gate, up, rows, cols, payload, scales, scale_cols, padded_scale_cols,
      scale_k_tiles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    write_error(error, error_len, cudaGetErrorString(err));
    return 3;
  }
  return 0;
}
