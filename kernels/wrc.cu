#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

__global__ void litmus_test(
  uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances) {

  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

#ifdef RELAXED
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
#elif defined(ACQUIRE)
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
#elif defined(RELEASE)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
#endif

#ifdef TB_012
#define X_0_SCOPE cuda::thread_scope_block
#define X_1_SCOPE cuda::thread_scope_block
#define Y_1_SCOPE cuda::thread_scope_block
#define X_2_SCOPE cuda::thread_scope_block
#define Y_2_SCOPE cuda::thread_scope_block

#elif defined(TB_0_12)
#define X_0_SCOPE cuda::thread_scope_device
#define X_1_SCOPE cuda::thread_scope_device
#define Y_1_SCOPE cuda::thread_scope_block
#define X_2_SCOPE cuda::thread_scope_device
#define Y_2_SCOPE cuda::thread_scope_block

#else
#define X_0_SCOPE cuda::thread_scope_device
#define X_1_SCOPE cuda::thread_scope_device
#define Y_1_SCOPE cuda::thread_scope_device
#define X_2_SCOPE cuda::thread_scope_device
#define Y_2_SCOPE cuda::thread_scope_device

#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    // defined for all three thread two memory locations tests
    THREE_THREAD_TWO_MEM_LOCATIONS();

    cuda::atomic<uint, X_0_SCOPE>* x_0_ptr = (cuda::atomic<uint, X_0_SCOPE>*) &test_locations[x_0];
    cuda::atomic<uint, X_1_SCOPE>* x_1_ptr = (cuda::atomic<uint, X_1_SCOPE>*) &test_locations[x_1];
    cuda::atomic<uint, Y_1_SCOPE>* y_1_ptr = (cuda::atomic<uint, Y_1_SCOPE>*) &test_locations[y_1];
    cuda::atomic<uint, X_2_SCOPE>* x_2_ptr = (cuda::atomic<uint, X_2_SCOPE>*) &test_locations[x_2];
    cuda::atomic<uint, Y_2_SCOPE>* y_2_ptr = (cuda::atomic<uint, Y_2_SCOPE>*) &test_locations[y_2];

    PRE_STRESS();

    if (id_0 != id_1 && id_1 != id_2 && id_0 != id_2) {

      // Thread 0
      x_0_ptr->store(1, cuda::memory_order_relaxed);

      // Thread 1
      uint r0 = x_1_ptr->load(thread_1_load);
      y_1_ptr->store(1, thread_1_store);

      // Thread 2
      uint r1 = y_2_ptr->load(thread_2_load);
      uint r2 = x_2_ptr->load(cuda::memory_order_relaxed);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_2].r1 = r1;
      read_results[wg_offset + id_2].r2 = r2;
    }
  }

  MEM_STRESS();
}

__global__ void check_results(
  uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && r1 == 1 && r2 == 1) {
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0) {
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1) {
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 0) {
    test_results->res3.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 1) {
    test_results->res4.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 0) {
    test_results->res5.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 1) {
    test_results->res6.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 0) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=1, r2=1 (seq): " << results->res0 << "\n";
    std::cout << "r0=0, r1=0, r2=0 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=1 (seq): " << results->res2 << "\n";
    std::cout << "r0=0, r1=1, r2=0 (seq): " << results->res3 << "\n";
    std::cout << "r0=0, r1=1, r2=1 (interleaved): " << results->res4 << "\n";
    std::cout << "r0=1, r1=0, r2=0 (seq): " << results->res5 << "\n";
    std::cout << "r0=1, r1=0, r2=1 (interleaved): " << results->res6 << "\n";
    std::cout << "r0=1, r1=1, r2=0 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

