// litmus.cuh
#ifndef LITMUS_CUH
#define LITMUS_CUH

#include <cuda_runtime.h>
#include <cuda/atomic>

#ifdef TB_0_1_2
#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint third_workgroup = stripe_workgroup(new_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = third_workgroup * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;
#elif defined(TB_01_2)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint id_1 = shuffled_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = new_workgroup * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;
#elif defined(TB_0_12)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint id_2 = new_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;
#elif defined(TB_02_1)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint id_2 = shuffled_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#elif defined(TB_012)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x; \
  uint id_0 = threadIdx.x; \
  uint id_1 = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = permute_id(id_1, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = shuffled_workgroup * blockDim.x;

#define RESULT_IDS() \
  uint total_ids = blockDim.x; \
  uint wg_offset = blockIdx.x * blockDim.x;
#else
// no inclusion
#endif

#ifdef TB_0_1_2_3
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_01_2_3)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1 = workgroup_1 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_01_23)
#define DEFINE_IDS()                                                                                                \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1 = workgroup_1 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0_1_23)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_02_1_3)
#define DEFINE_IDS()                                                                                                \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = shuffled_workgroup; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_02_13)
#define DEFINE_IDS()                                                                                                \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = shuffled_workgroup; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = workgroup_1; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0_2_13)
#define DEFINE_IDS()                                                                                                \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_1; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_03_1_2)
#define DEFINE_IDS()                                                                                                \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = shuffled_workgroup; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_03_12)
#define DEFINE_IDS()                                                                                                \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = workgroup_1; \
  uint id_2 = workgroup_1 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = shuffled_workgroup; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0_12_3)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = workgroup_1; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_0_123)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = workgroup_1; \
  uint id_2_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = workgroup_2 * blockDim.x + id_2_local; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(id_2_local, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_012_3)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_1 = workgroup_1 * blockDim.x + id_1_local; \
  uint workgroup_2 = workgroup_1; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(id_1_local, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_023_1)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);   \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = shuffled_workgroup; \
  uint id_2_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = workgroup_2 * blockDim.x + id_2_local; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(id_2_local, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_013_2)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups);                                           \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                   \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_1 = workgroup_1 * blockDim.x + id_1_local; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_1; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(id_1_local, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0123)
#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x; \
  uint id_0 = threadIdx.x; \
  uint id_1 = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = permute_id(id_1, kernel_params->permute_thread, blockDim.x); \
  uint id_3 = permute_id(id_2, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = shuffled_workgroup * blockDim.x;
#else
// no inclusion
#endif

#define THREE_THREAD_TWO_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2; \
  uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2; \
  uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 2; \
  uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x; \
  test_instances[id_0].t0 = t_id; \
  test_instances[id_1].t1 = t_id; \
  test_instances[id_2].t2 = t_id; \
  test_instances[id_0].x = x_0; \
  test_instances[id_0].y = y_0;

#define THREE_THREAD_THREE_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 3; \
  uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids); \
  uint y_0 = (wg_offset + permute_id_0) * kernel_params->mem_stride * 3 + kernel_params->mem_offset; \
  uint z_0 = (wg_offset + permute_id(permute_id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset; \
  uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids); \
  uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 3 + kernel_params->mem_offset; \
  uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset; \
  uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 3; \
  uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids); \
  uint z_2 = (wg_offset + permute_id(permute_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset; \
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x; \
  test_instances[id_0].t0 = t_id; \
  test_instances[id_1].t1 = t_id; \
  test_instances[id_2].t2 = t_id; \
  test_instances[id_0].x = x_0; \
  test_instances[id_0].y = y_0; \
  test_instances[id_0].z = z_0;

#define PRE_STRESS() \
  if (kernel_params->pre_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern); \
  } \
  if (kernel_params->barrier) { \
    spin(barrier, blockDim.x * kernel_params->testing_workgroups); \
  }

#define MEM_STRESS() \
  else if (kernel_params->mem_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->pre_stress_iterations); \
  }

typedef struct {
  uint r0;
  uint r1;
  uint r2;
  uint r3;
} ReadResults;

typedef struct {
  uint t0;
  uint t1;
  uint t2;
  uint t3;
  uint x;
  uint y;
  uint z;
} TestInstance;

typedef struct {
  cuda::atomic<uint, cuda::thread_scope_device> res0; // up to 16 combinations of valid test results in the implemented tests
  cuda::atomic<uint, cuda::thread_scope_device> res1;
  cuda::atomic<uint, cuda::thread_scope_device> res2;
  cuda::atomic<uint, cuda::thread_scope_device> res3;
  cuda::atomic<uint, cuda::thread_scope_device> res4;
  cuda::atomic<uint, cuda::thread_scope_device> res5;
  cuda::atomic<uint, cuda::thread_scope_device> res6;
  cuda::atomic<uint, cuda::thread_scope_device> res7;
  cuda::atomic<uint, cuda::thread_scope_device> res8;
  cuda::atomic<uint, cuda::thread_scope_device> res9;
  cuda::atomic<uint, cuda::thread_scope_device> res10;
  cuda::atomic<uint, cuda::thread_scope_device> res11;
  cuda::atomic<uint, cuda::thread_scope_device> res12;
  cuda::atomic<uint, cuda::thread_scope_device> res13;
  cuda::atomic<uint, cuda::thread_scope_device> res14;
  cuda::atomic<uint, cuda::thread_scope_device> res15;
  cuda::atomic<uint, cuda::thread_scope_device> weak; // this is the weak behavior we are looking for
  cuda::atomic<uint, cuda::thread_scope_device> na; // some threads don't execute the test if thread ids clash
  cuda::atomic<uint, cuda::thread_scope_device> other; // this should always be 0
} TestResults;


typedef struct {
  bool barrier;
  bool mem_stress;
  int mem_stress_iterations;
  int mem_stress_pattern;
  bool pre_stress;
  int pre_stress_iterations;
  int pre_stress_pattern;
  int permute_thread;
  int permute_location;
  int testing_workgroups;
  int mem_stride;
  int mem_offset;
} KernelParams;

__global__ void litmus_test(
  uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances);

__global__ void check_results(
  uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak);

int host_check_results(TestResults* results, bool print);

#endif
