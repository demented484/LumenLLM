// CUTLASS NVFP4 Grouped GEMM for SM120 (consumer Blackwell, RTX 5070 Ti).
//
// Mirrors CUTLASS example 79d (`79d_blackwell_geforce_nvfp4_grouped_gemm`)
// but with:
//   * ElementD = float (matches our downstream GeGLU which reads f32 inputs).
//   * No epilogue scale-factor generation; plain `alpha * acc + beta * C`
//     epilogue with a single per-group alpha (== expert `output_scale`).
//   * No host_problem_shapes_available path — the caller uploads
//     `problem_sizes`, `ptr_A`, `ptr_B`, `ptr_SFA`, `ptr_SFB`, `ptr_D`,
//     `stride_A`, `stride_B`, `stride_D`, `layout_SFA`, `layout_SFB`,
//     `alpha_ptr` (one float per group) directly.
//
// Build harness: this file is compiled as a second translation unit by
// build.rs. It is enabled only when CUTLASS_ARCH_MMA_SM120_SUPPORTED.
//
// Activation enabled via env AEGIS_CUTLASS_NVFP4_GROUPED=1 in the MoE
// prefill dispatcher; the C++ symbols here are unconditionally exported
// so a runtime can_implement() check decides eligibility per call.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include <cuda_fp8.h>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace aegis_cutlass_bridge_moe {

// ─────────────────────────────────────────────────────────────────────────────
// Device-side helpers shared with the dense path (intentionally re-declared
// here so this TU is self-contained — both TUs are linked into the same
// archive but ODR-isolated under their own namespaces). The SFA swizzle
// layout matches `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA` for
// (Mtile=128, K-scale-tile=4) which is the layout produced by the CUTLASS
// instantiation in this file (ThreadBlockShape = _128,_128,_128).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

__device__ __forceinline__ unsigned moe_best_e2m1(float value, float scale) {
  if (!(scale > 0.0f)) {
    return 0;
  }
  float scaled = value / scale;
  if (!isfinite(scaled)) {
    return 0;
  }
  float mag = fabsf(scaled);
  unsigned code = 0;
  if (mag <= 0.25f) {
    code = 0;
  } else if (mag <= 0.75f) {
    code = 1;
  } else if (mag <= 1.25f) {
    code = 2;
  } else if (mag <= 1.75f) {
    code = 3;
  } else if (mag <= 2.5f) {
    code = 4;
  } else if (mag <= 3.5f) {
    code = 5;
  } else if (mag <= 5.0f) {
    code = 6;
  } else {
    code = 7;
  }
  return scaled < 0.0f && code != 0 ? (code | 0x8u) : code;
}

// Layout: padded_rows × padded_scale_cols, tiled in 128×4 blocks of
// 512 bytes each. row_in_blob is the row inside this per-expert SFA
// blob (so >= 0 and < padded_rows). scale_col in [0, padded_scale_cols).
__device__ __forceinline__ size_t moe_swizzled_scale_offset(
    int row_in_blob, int scale_col, int scale_k_tiles) {
  int m_tile_idx = row_in_blob >> 7;
  int outer_m_idx = row_in_blob & 31;
  int inner_m_idx = (row_in_blob >> 5) & 3;
  int k_tile_idx = scale_col >> 2;
  int inner_k_idx = scale_col & 3;
  return (static_cast<size_t>(m_tile_idx) * scale_k_tiles + k_tile_idx) * 512u +
         static_cast<size_t>(outer_m_idx) * 16u +
         static_cast<size_t>(inner_m_idx) * 4u + inner_k_idx;
}

}  // namespace

// Per-expert NVFP4 quantizer for grouped MoE inputs.
//
// Inputs:
//   `input`              : permuted activations, [total_tokens, cols], row-major f32.
//   `token_offsets`      : per-expert prefix-sum of token counts, length = num_groups+1.
//   `payload_offsets`    : per-expert offset (bytes) into `payload_out` buffer,
//                          length = num_groups. payload_out[payload_offsets[g] +
//                          row_in_g * (cols/2) + .. ] holds expert g's packed nibbles.
//   `sfa_offsets`        : per-expert offset (bytes) into `sfa_out` buffer,
//                          length = num_groups. sfa_out[sfa_offsets[g] + ..]
//                          holds expert g's swizzled SFA blob of size
//                          padded_rows_g * padded_scale_cols bytes.
// All offsets are device-resident (uploaded by the caller). The kernel
// processes one (group, row, scale_col) tile per thread.
//
// Grid: (max_padded_scale_cols / threads_per_block, max_padded_rows_per_group, num_groups)
// We over-launch and gate on per-group padded_rows / padded_scale_cols read
// from token_offsets. This keeps the launch parameters static.
__global__ void quantize_grouped_f32_to_e2m1_ue4m3_kernel(
    const float* __restrict__ input, int cols, int scale_cols,
    int padded_scale_cols, int scale_k_tiles, int num_groups,
    const uint32_t* __restrict__ token_offsets,
    const uint64_t* __restrict__ payload_offsets,
    const uint64_t* __restrict__ sfa_offsets, uint8_t* __restrict__ payload_out,
    uint8_t* __restrict__ sfa_out) {
  int group = blockIdx.z;
  if (group >= num_groups) return;
  int row_in_group = blockIdx.y;
  int scale_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (scale_col >= padded_scale_cols) return;

  uint32_t row_start = token_offsets[group];
  uint32_t row_end = token_offsets[group + 1];
  int rows_in_group = static_cast<int>(row_end - row_start);
  int padded_rows = ((rows_in_group + 127) / 128) * 128;
  if (row_in_group >= padded_rows) return;

  uint64_t payload_off = payload_offsets[group];
  uint64_t sfa_off = sfa_offsets[group];
  uint8_t* group_payload = payload_out + payload_off;
  uint8_t* group_sfa = sfa_out + sfa_off;

  if (row_in_group >= rows_in_group || scale_col >= scale_cols) {
    group_sfa[moe_swizzled_scale_offset(row_in_group, scale_col,
                                        scale_k_tiles)] = 0;
    return;
  }

  size_t absolute_row = static_cast<size_t>(row_start) + row_in_group;
  int col_base = scale_col * 16;
  float values[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float value = input[absolute_row * cols + col_base + i];
    values[i] = value;
    amax = fmaxf(amax, fabsf(value));
  }

  float scale = (amax > 0.0f) ? (amax / 6.0f) : 0.0f;
  __nv_fp8_e4m3 encoded_scale(scale);
  uint8_t scale_byte = reinterpret_cast<uint8_t&>(encoded_scale);
  group_sfa[moe_swizzled_scale_offset(row_in_group, scale_col, scale_k_tiles)] =
      scale_byte;
  float decoded_scale = static_cast<float>(encoded_scale);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    unsigned lo = moe_best_e2m1(values[pair * 2], decoded_scale);
    unsigned hi = moe_best_e2m1(values[pair * 2 + 1], decoded_scale);
    group_payload[static_cast<size_t>(row_in_group) * (cols / 2) +
                  scale_col * 8 + pair] = static_cast<uint8_t>(lo | (hi << 4));
  }
}

// Per-expert weight-scale swizzler. Reads row-major scale bytes from a
// staged bulk buffer (one per expert, as produced by the existing host
// load + H2D path) and writes them into a swizzled per-expert SFB blob.
//
// Inputs:
//   `src`              : row-major scales for each expert concatenated in `src`,
//                        indexed by `src_offsets[g]` (in bytes).
//   `src_rows_per_g`   : rows per expert (== N_g, the projection's output rows).
//   `src_cols`         : == K_g / 16 (== scale_cols, same across experts since
//                        K is shared).
//   `dst_offsets`      : per-expert offsets (in bytes) into the swizzled
//                        destination buffer.
//   `padded_scale_cols`: scale_cols rounded up to multiple of 4.
//   `scale_k_tiles`    : padded_scale_cols / 4.
//   `padded_rows`      : src_rows_per_g rounded up to multiple of 128
//                        (same for every expert since N is shared).
//   `num_groups`       : number of experts.
//   `dst`              : output buffer.
__global__ void swizzle_grouped_scales_kernel(
    const uint8_t* __restrict__ src, int src_rows_per_g, int src_cols,
    int padded_scale_cols, int scale_k_tiles, int padded_rows, int num_groups,
    const uint64_t* __restrict__ src_offsets,
    const uint64_t* __restrict__ dst_offsets, uint8_t* __restrict__ dst) {
  int group = blockIdx.z;
  if (group >= num_groups) return;
  int row = blockIdx.y;
  int scale_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= padded_rows || scale_col >= padded_scale_cols) return;

  uint64_t src_off = src_offsets[group];
  uint64_t dst_off = dst_offsets[group];
  const uint8_t* src_grp = src + src_off;
  uint8_t* dst_grp = dst + dst_off;

  uint8_t value = 0;
  if (row < src_rows_per_g && scale_col < src_cols) {
    value = src_grp[static_cast<size_t>(row) * src_cols + scale_col];
  }
  dst_grp[moe_swizzled_scale_offset(row, scale_col, scale_k_tiles)] = value;
}

extern "C" int aegis_cutlass_moe_nvfp4_quantize_input_grouped(
    const float* input, int cols, int num_groups,
    const uint32_t* token_offsets_device, const uint64_t* payload_offsets_device,
    const uint64_t* sfa_offsets_device, int max_padded_rows_per_group,
    uint8_t* payload_out, uint8_t* sfa_out, void* stream) {
  if (input == nullptr || token_offsets_device == nullptr ||
      payload_offsets_device == nullptr || sfa_offsets_device == nullptr ||
      payload_out == nullptr || sfa_out == nullptr) {
    return 1;
  }
  if (num_groups <= 0 || cols <= 0 || (cols % 32) != 0) {
    return 2;
  }
  int scale_cols = cols / 16;
  int padded_scale_cols = ((scale_cols + 3) / 4) * 4;
  int scale_k_tiles = padded_scale_cols / 4;
  dim3 block(128, 1, 1);
  dim3 grid((padded_scale_cols + block.x - 1) / block.x,
            max_padded_rows_per_group, num_groups);
  quantize_grouped_f32_to_e2m1_ue4m3_kernel<<<grid, block, 0,
                                              reinterpret_cast<cudaStream_t>(
                                                  stream)>>>(
      input, cols, scale_cols, padded_scale_cols, scale_k_tiles, num_groups,
      token_offsets_device, payload_offsets_device, sfa_offsets_device,
      payload_out, sfa_out);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : 3;
}

extern "C" int aegis_cutlass_moe_nvfp4_swizzle_weight_scales_grouped(
    const uint8_t* src, int rows_per_group, int src_cols, int num_groups,
    const uint64_t* src_offsets_device, const uint64_t* dst_offsets_device,
    uint8_t* dst, void* stream) {
  if (src == nullptr || src_offsets_device == nullptr ||
      dst_offsets_device == nullptr || dst == nullptr) {
    return 1;
  }
  if (num_groups <= 0 || rows_per_group <= 0 || src_cols <= 0) {
    return 2;
  }
  int padded_scale_cols = ((src_cols + 3) / 4) * 4;
  int scale_k_tiles = padded_scale_cols / 4;
  int padded_rows = ((rows_per_group + 127) / 128) * 128;
  dim3 block(128, 1, 1);
  dim3 grid((padded_scale_cols + block.x - 1) / block.x, padded_rows,
            num_groups);
  swizzle_grouped_scales_kernel<<<grid, block, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
      src, rows_per_group, src_cols, padded_scale_cols, scale_k_tiles,
      padded_rows, num_groups, src_offsets_device, dst_offsets_device, dst);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : 3;
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;

using ElementA = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
static constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
static constexpr int AlignmentB = 32;

// Output: float, identical to our home-rolled grouped GEMM.
using ElementD = float;
using ElementC = float;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Cluster fixed to 1x1x1 for SM120 (consumer Blackwell).
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag*, AlignmentC, ElementD,
        LayoutCTag*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB,
        LayoutBTag*, AlignmentB, ElementAccumulator, ThreadBlockShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape,
                                                       CollectiveMainloop,
                                                       CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Per-group Internal* types (one entry per expert in the device arrays).
using InternalStrideA = typename Gemm::GemmKernel::InternalStrideA;
using InternalStrideB = typename Gemm::GemmKernel::InternalStrideB;
using InternalStrideC = typename Gemm::GemmKernel::InternalStrideC;
using InternalStrideD = typename Gemm::GemmKernel::InternalStrideD;
using InternalLayoutSFA =
    typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using InternalLayoutSFB =
    typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Host helper exposed via the C entry points: given a problem shape
// (m, n, k), produce the CUTLASS-internal stride/layout values that the
// caller must upload to device-side arrays.
extern "C" int aegis_cutlass_moe_nvfp4_compute_strides_sm120(
    int m, int n, int k, void* stride_a_out, void* stride_b_out,
    void* stride_d_out, void* layout_sfa_out, void* layout_sfb_out) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return 1;
  }
  if ((k % 32) != 0 || (n % 32) != 0) {
    return 2;
  }
  auto stride_a =
      cutlass::make_cute_packed_stride(InternalStrideA{}, {m, k, 1});
  auto stride_b =
      cutlass::make_cute_packed_stride(InternalStrideB{}, {n, k, 1});
  auto stride_d =
      cutlass::make_cute_packed_stride(InternalStrideD{}, {m, n, 1});
  auto layout_sfa =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  // Copy out raw bytes; the Rust side carries them as opaque blobs of the
  // sizes returned by aegis_cutlass_moe_nvfp4_stride_sizes_sm120.
  if (stride_a_out) {
    std::memcpy(stride_a_out, &stride_a, sizeof(stride_a));
  }
  if (stride_b_out) {
    std::memcpy(stride_b_out, &stride_b, sizeof(stride_b));
  }
  if (stride_d_out) {
    std::memcpy(stride_d_out, &stride_d, sizeof(stride_d));
  }
  if (layout_sfa_out) {
    std::memcpy(layout_sfa_out, &layout_sfa, sizeof(layout_sfa));
  }
  if (layout_sfb_out) {
    std::memcpy(layout_sfb_out, &layout_sfb, sizeof(layout_sfb));
  }
  return 0;
}

extern "C" int aegis_cutlass_moe_nvfp4_stride_sizes_sm120(
    size_t* stride_a_bytes, size_t* stride_b_bytes, size_t* stride_d_bytes,
    size_t* layout_sfa_bytes, size_t* layout_sfb_bytes,
    size_t* problem_shape_bytes) {
  if (stride_a_bytes) *stride_a_bytes = sizeof(InternalStrideA);
  if (stride_b_bytes) *stride_b_bytes = sizeof(InternalStrideB);
  if (stride_d_bytes) *stride_d_bytes = sizeof(InternalStrideD);
  if (layout_sfa_bytes) *layout_sfa_bytes = sizeof(InternalLayoutSFA);
  if (layout_sfb_bytes) *layout_sfb_bytes = sizeof(InternalLayoutSFB);
  if (problem_shape_bytes)
    *problem_shape_bytes =
        sizeof(typename ProblemShape::UnderlyingProblemShape);
  return 0;
}

// Given a per-group problem (M_g, N_g, K_g), return the per-group
// SFA/SFB tensor sizes in bytes (== cosize of the swizzled layout
// times sizeof(ElementSF)). This is what the caller must allocate
// for that group's scale-factor blob inside its bulk buffer.
//
// SFA depends on (M_g, K_g) only; SFB on (N_g, K_g) only — but we
// take all three for symmetry with `compute_strides`.
extern "C" int aegis_cutlass_moe_nvfp4_sfa_sfb_bytes_sm120(
    int m, int n, int k, size_t* sfa_bytes_out, size_t* sfb_bytes_out) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return 1;
  }
  if ((k % 32) != 0 || (n % 32) != 0) {
    return 2;
  }
  auto layout_sfa =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  if (sfa_bytes_out) {
    *sfa_bytes_out = static_cast<size_t>(cute::size(layout_sfa)) *
                     sizeof(ElementSF);
  }
  if (sfb_bytes_out) {
    *sfb_bytes_out = static_cast<size_t>(cute::size(layout_sfb)) *
                     sizeof(ElementSF);
  }
  return 0;
}

static void write_error(char* error, size_t error_len, const char* message) {
  if (error == nullptr || error_len == 0) {
    return;
  }
  std::snprintf(error, error_len, "%s", message);
}

// Workspace query for the grouped GEMM. Callers must allocate at least
// the returned number of bytes plus space for the per-group device-side
// arrays (which they manage themselves).
extern "C" int aegis_cutlass_moe_nvfp4_workspace_size_sm120(
    int num_groups, void* device_problem_sizes,
    const void* host_problem_sizes_or_null, size_t* workspace_bytes,
    char* error, size_t error_len) {
  if (num_groups <= 0 || device_problem_sizes == nullptr ||
      workspace_bytes == nullptr) {
    write_error(error, error_len,
                "CUTLASS MoE NVFP4 workspace: invalid arguments");
    return 1;
  }
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  // Probe arguments: pointers/strides may be null when only the
  // workspace size is needed; CUTLASS only reads the group count and
  // problem-size array sizes.
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups,
       static_cast<typename ProblemShape::UnderlyingProblemShape*>(
           device_problem_sizes),
       static_cast<const typename ProblemShape::UnderlyingProblemShape*>(
           host_problem_sizes_or_null)},
      {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
      {{}, nullptr, nullptr, nullptr, nullptr},
      hw_info,
      scheduler};
  Gemm gemm;
  *workspace_bytes = Gemm::get_workspace_size(arguments);
  return 0;
}

// Run the grouped NVFP4 GEMM. All device-side per-group arrays are
// caller-owned and uploaded once. `alpha_device` is a per-group array
// of f32 values (typically each expert's `output_scale`). `beta` is
// 0 (no C input).
extern "C" int aegis_cutlass_moe_nvfp4_grouped_gemm_sm120(
    int num_groups, void* device_problem_sizes,
    const void* host_problem_sizes_or_null,
    void** ptr_A,         // device array of A pointers, num_groups entries
    void** ptr_B,         // device array of B pointers
    void** ptr_SFA,       // device array of SFA pointers
    void** ptr_SFB,       // device array of SFB pointers
    void** ptr_D,         // device array of D pointers
    void* stride_A,       // device array of InternalStrideA, num_groups entries
    void* stride_B,       // device array of InternalStrideB
    void* stride_D,       // device array of InternalStrideD
    void* layout_SFA,     // device array of InternalLayoutSFA
    void* layout_SFB,     // device array of InternalLayoutSFB
    float** alpha_device, // device array of f32* (1 per group, points to a
                          // single per-group output_scale)
    void* workspace, size_t workspace_bytes, void* stream, char* error,
    size_t error_len) {
  if (num_groups <= 0) {
    write_error(error, error_len,
                "CUTLASS MoE NVFP4 grouped GEMM: num_groups <= 0");
    return 1;
  }
  if (device_problem_sizes == nullptr || ptr_A == nullptr ||
      ptr_B == nullptr || ptr_SFA == nullptr || ptr_SFB == nullptr ||
      ptr_D == nullptr) {
    write_error(error, error_len,
                "CUTLASS MoE NVFP4 grouped GEMM: null pointer");
    return 2;
  }
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  using FusionArgs =
      typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;
  FusionArgs fusion_args{};
  fusion_args.alpha = 0;
  fusion_args.alpha_ptr_array = alpha_device;
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  fusion_args.beta = 0;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  // Mainloop Arguments field order (sm120_blockscaled_mma_array_tma.hpp:
  // 285-293): ptr_A, dA, ptr_B, dB, ptr_SFA, layout_SFA, ptr_SFB,
  // layout_SFB. Each stride/layout member is a *pointer* to a per-group
  // device array of the Internal* type (grouped GEMM specialization).
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups,
       static_cast<typename ProblemShape::UnderlyingProblemShape*>(
           device_problem_sizes),
       static_cast<const typename ProblemShape::UnderlyingProblemShape*>(
           host_problem_sizes_or_null)},
      {const_cast<typename Gemm::ElementA const**>(
           reinterpret_cast<typename Gemm::ElementA**>(ptr_A)),
       static_cast<InternalStrideA*>(stride_A),
       const_cast<typename Gemm::ElementB const**>(
           reinterpret_cast<typename Gemm::ElementB**>(ptr_B)),
       static_cast<InternalStrideB*>(stride_B),
       const_cast<ElementSF const**>(
           reinterpret_cast<ElementSF**>(ptr_SFA)),
       static_cast<InternalLayoutSFA*>(layout_SFA),
       const_cast<ElementSF const**>(
           reinterpret_cast<ElementSF**>(ptr_SFB)),
       static_cast<InternalLayoutSFB*>(layout_SFB)},
      {fusion_args,
       /*ptr_C=*/nullptr,
       /*stride_C=*/static_cast<InternalStrideC*>(nullptr),
       reinterpret_cast<typename Gemm::EpilogueOutputOp::ElementOutput**>(
           ptr_D),
       static_cast<InternalStrideD*>(stride_D)},
      hw_info,
      scheduler};

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 10 + static_cast<int>(status);
  }
  size_t required_workspace = Gemm::get_workspace_size(arguments);
  if (workspace_bytes < required_workspace) {
    write_error(error, error_len,
                "CUTLASS MoE NVFP4 grouped GEMM workspace too small");
    return 20;
  }
  status = gemm.initialize(arguments, workspace,
                           reinterpret_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 30 + static_cast<int>(status);
  }
  status = gemm.run(arguments, workspace,
                    reinterpret_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    write_error(error, error_len, cutlassGetStatusString(status));
    return 40 + static_cast<int>(status);
  }
  return 0;
}

#else  // !CUTLASS_ARCH_MMA_SM120_SUPPORTED

extern "C" int aegis_cutlass_moe_nvfp4_stride_sizes_sm120(
    size_t* stride_a_bytes, size_t* stride_b_bytes, size_t* stride_d_bytes,
    size_t* layout_sfa_bytes, size_t* layout_sfb_bytes,
    size_t* problem_shape_bytes) {
  (void)stride_a_bytes;
  (void)stride_b_bytes;
  (void)stride_d_bytes;
  (void)layout_sfa_bytes;
  (void)layout_sfb_bytes;
  (void)problem_shape_bytes;
  return 1000;
}

extern "C" int aegis_cutlass_moe_nvfp4_compute_strides_sm120(
    int m, int n, int k, void* stride_a_out, void* stride_b_out,
    void* stride_d_out, void* layout_sfa_out, void* layout_sfb_out) {
  (void)m;
  (void)n;
  (void)k;
  (void)stride_a_out;
  (void)stride_b_out;
  (void)stride_d_out;
  (void)layout_sfa_out;
  (void)layout_sfb_out;
  return 1000;
}

extern "C" int aegis_cutlass_moe_nvfp4_sfa_sfb_bytes_sm120(
    int m, int n, int k, size_t* sfa_bytes_out, size_t* sfb_bytes_out) {
  (void)m;
  (void)n;
  (void)k;
  (void)sfa_bytes_out;
  (void)sfb_bytes_out;
  return 1000;
}

extern "C" int aegis_cutlass_moe_nvfp4_workspace_size_sm120(
    int num_groups, void* device_problem_sizes,
    const void* host_problem_sizes_or_null, size_t* workspace_bytes,
    char* error, size_t error_len) {
  (void)num_groups;
  (void)device_problem_sizes;
  (void)host_problem_sizes_or_null;
  (void)workspace_bytes;
  if (error && error_len) {
    std::snprintf(error, error_len,
                  "CUTLASS SM120 support not compiled (MoE NVFP4 grouped GEMM)");
  }
  return 1000;
}

extern "C" int aegis_cutlass_moe_nvfp4_grouped_gemm_sm120(
    int num_groups, void* device_problem_sizes,
    const void* host_problem_sizes_or_null, void** ptr_A, void** ptr_B,
    void** ptr_SFA, void** ptr_SFB, void** ptr_D, void* stride_A,
    void* stride_B, void* stride_D, void* layout_SFA, void* layout_SFB,
    float** alpha_device, void* workspace, size_t workspace_bytes,
    void* stream, char* error, size_t error_len) {
  (void)num_groups;
  (void)device_problem_sizes;
  (void)host_problem_sizes_or_null;
  (void)ptr_A;
  (void)ptr_B;
  (void)ptr_SFA;
  (void)ptr_SFB;
  (void)ptr_D;
  (void)stride_A;
  (void)stride_B;
  (void)stride_D;
  (void)layout_SFA;
  (void)layout_SFB;
  (void)alpha_device;
  (void)workspace;
  (void)workspace_bytes;
  (void)stream;
  if (error && error_len) {
    std::snprintf(error, error_len,
                  "CUTLASS SM120 support not compiled (MoE NVFP4 grouped GEMM)");
  }
  return 1000;
}

#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED

}  // namespace aegis_cutlass_bridge_moe
