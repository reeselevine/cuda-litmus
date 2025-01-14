#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

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

#ifdef ACQ_REL
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(ACQ_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(REL_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(REL_REL)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(RELAXED)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(BOTH_FENCE)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_1_FENCE_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2() 
#elif defined(THREAD_1_FENCE_REL)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2() 
#elif defined(THREAD_2_FENCE_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1() 
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_2_FENCE_REL)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1() 
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#else
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed; // default to all relaxed
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2()
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    // defined for all three thread two memory locations tests
    THREE_THREAD_TWO_MEM_LOCATIONS();

    PRE_STRESS();

    if (id_0 != id_1 && id_1 != id_2 && id_0 != id_2) {

      // Thread 0
      test_locations[x_0].store(2, cuda::memory_order_relaxed);

      // Thread 1
      uint r0 = test_locations[x_1].load(thread_1_load);
      FENCE_1()
      test_locations[y_1].store(1, thread_1_store);

      // Thread 2
      uint r1 = test_locations[y_2].load(thread_2_load);
      FENCE_2()
      test_locations[x_2].store(1, thread_2_store);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_2].r1 = r1;
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
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];

  if (r0 == 1 && r1 == 1) {
    test_results->res0.fetch_add(1); // this is actually a load buffer weak behavior
  }
  else if (r0 == 2 && r1 == 1 && x == 2) { // this is the non-mca weak behavior
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  } 
  else if (r0 <= 2 && r1 <= 1 && (x == 2 || x == 1)) { // catch all for other sequential/interleaved behaviors
    test_results->res1.fetch_add(1);
  }
  else if (x == 0) { // thread skipped
    test_results->na.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0 <= 2, r1 <= 1, x <= 2 (seq/interleaved): " << results->res1 << "\n";
    std::cout << "r0=1, r1=1, x = (1 || 2) (lb weak): " << results->res0 << "\n";
    std::cout << "r0=2, r1=1, x=2 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}
