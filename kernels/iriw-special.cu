#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

// iriw thread mappings
// thread 0: write x
// thread 1: read x, read y
// thread 2: write y
// thread 3: read y, read x

__global__ void litmus_test(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances) {
  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2;
    uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2;
    uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 2;
    uint y_3 = (wg_offset + permute_id(id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    // Save threads and memory locations involved in a test instance
    uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
    test_instances[id_0].t0 = t_id;
    test_instances[id_1].t1 = t_id;
    test_instances[id_2].t2 = t_id;
    test_instances[id_3].t3 = t_id;
    test_instances[id_0].x = x_0;
    test_instances[id_0].y = y_0;

    PRE_STRESS();

    if (id_0 != id_1 && id_0 != id_2 && id_0 != id_3 && id_1 != id_2 && id_1 != id_3 && id_2 != id_3) {

      test_locations[x_0].store(1, cuda::memory_order_release); // write x

      uint r0 = test_locations[x_1].load(cuda::memory_order_relaxed); // read x
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);
      uint r1 = test_locations[y_1].load(cuda::memory_order_acquire); // read y

      test_locations[y_2].store(1, cuda::memory_order_release); // write y

      uint r2 = test_locations[y_3].load(cuda::memory_order_relaxed); // read y
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);
      uint r3 = test_locations[x_3].load(cuda::memory_order_acquire); // read x

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
      read_results[wg_offset + id_3].r2 = r2;
      read_results[wg_offset + id_3].r3 = r3;
    }
  }
  MEM_STRESS();
}

__global__ void check_results(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint r3 = read_results[id_0].r3;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0 && r3 == 0) { // both observers run first
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 1 && r3 == 1) { // both observers run last
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1 && r3 == 1) { // first observer runs first
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 0 && r3 == 0) { // second observer runs first
    test_results->res3.fetch_add(1);
  }
  else if (r0 == r1 && r2 != r3) { // second observer interleaved
    test_results->res4.fetch_add(1);
  }
  else if (r0 != r1 && r2 == r3) { // first observer interleaved
    test_results->res5.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 0 && r3 == 1) { // both interleaved
    test_results->res6.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 1 && r3 == 0) { // both interleaved
    test_results->res7.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 0 && r3 == 1) { // both interleaved
    test_results->res8.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) { // observer threads see x/y in different orders
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=0, r2=0, r3=0 (seq): " << results->res0 << "\n";
    std::cout << "r0=1, r1=1, r2=1, r3=1 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=1, r3=1 (seq): " << results->res2 << "\n";
    std::cout << "r0=1, r1=1, r2=0, r3=0 (seq): " << results->res3 << "\n";
    std::cout << "r0 == r1, r2 != r3 (seq/interleaved): " << results->res4 << "\n";
    std::cout << "r0 != r1, r2 == r3 (interleaved/seq): " << results->res5 << "\n";
    std::cout << "r0=0, r1=1, r2=0, r3=1 (interleaved): " << results->res6 << "\n";
    std::cout << "r0=0, r1=1, r2=1, r3=0 (interleaved): " << results->res7 << "\n";
    std::cout << "r0=1, r1=0, r2=0, r3=1 (interleaved): " << results->res8 << "\n";
    std::cout << "r0=1, r1=0, r2=1, r3=0 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

