#include <iostream>
#include "litmus.cuh"
#include "functions.cuh"

__global__ void litmus_test(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params) {

  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {
    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);

    uint id_1 = new_workgroup * blockDim.x + threadIdx.x;
    uint third_workgroup = stripe_workgroup(new_workgroup, threadIdx.x, kernel_params->testing_workgroups);

    uint id_2 = third_workgroup * blockDim.x + threadIdx.x;
    uint x_0 = id_0 * kernel_params->mem_stride * 3;
    uint y_0 = permute_id(id_0, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = permute_id_1 * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint z_1 = permute_id(permute_id_1, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;
    uint x_2 = id_2 * kernel_params->mem_stride * 3;
    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint z_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    if (kernel_params->pre_stress) {
      do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern);
    }
    if (kernel_params->barrier) {
      spin(barrier, blockDim.x * kernel_params->testing_workgroups);
    }

    if (id_1 != id_2) {

      // Thread 0
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      test_locations[y_0].store(1, cuda::memory_order_relaxed);

      // Thread 1
      uint r0 = test_locations[y_1].load(cuda::memory_order_relaxed);
      test_locations[z_1].store(1, cuda::memory_order_relaxed);

      // Thread 2
      uint r1 = test_locations[z_2].load(cuda::memory_order_relaxed);
      uint r2 = test_locations[x_2].load(cuda::memory_order_relaxed);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_2].r1 = r1;
      read_results[id_2].r2 = r2;
    }
  }
  else if (kernel_params->mem_stress) {
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->pre_stress_iterations);
  }
}

__global__ void check_results(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;

  if (r0 == 1 && r1 == 1 && r2 == 1) {
    test_results->seq0.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0) {
    test_results->seq1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1) {
    test_results->seq2.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 0) {
    test_results->seq3.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 1) {
    test_results->interleaved0.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 0) {
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=1, r2=1 (seq): " << results->seq0 << "\n";
    std::cout << "r0=0, r1=0, r2=0 (seq): " << results->seq1 << "\n";
    std::cout << "r0=0, r1=0, r2=1 (seq): " << results->seq2 << "\n";
    std::cout << "r0=1, r1=0, r2=0 (seq): " << results->seq3 << "\n";
    std::cout << "r0=1, r1=0, r2=1 (interleaved): " << results->interleaved0 << "\n";
    std::cout << "r0=1, r1=1, r2=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";

  }
  return results->weak;
}
