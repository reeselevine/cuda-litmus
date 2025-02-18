#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

#ifdef VOLATILE
    #define STORE_X() test_locations[x_0] = 1;
    #define STORE_Y() test_locations[x_0] = 1;
    #define LOAD_Y() uint r0 = test_locations[y_1];
    #define LOAD_X() uint r1 = test_locations[x_1];
#elif defined(RMW)
    #define STORE_X() atomicAdd(&test_locations[x_0], 1);
    #define STORE_Y() atomicExch(&test_locations[x_0], 1);
    #define LOAD_Y() uint r0 = atomicCAS(&test_locations[y_1], 1, 0);
    #define LOAD_X() uint r1 = atomicAdd(&test_locations[x_1], 0);
#endif

__global__ void litmus_test(
  d_uint_type* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances) {
  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);
    uint id_1 = workgroup_1 * blockDim.x + threadIdx.x;

    uint x_0 = id_0 * kernel_params->mem_stride * 2;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint y_0 = permute_id_0 * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    uint x_1 = id_1 * kernel_params->mem_stride * 2;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = permute_id_1 * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    PRE_STRESS();

    if (id_0 != id_1) {

      STORE_X()
      __threadfence_block();
      STORE_Y()

      LOAD_Y()
      __threadfence_block();
      LOAD_X()

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_1].r1 = r1;
    }
  }
  MEM_STRESS();
}

__global__ void check_results(
  d_uint_type* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && r1 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
  }
  return results->weak;
}

