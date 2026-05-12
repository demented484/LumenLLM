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
