#include <iostream>
#include <cuda_runtime.h>
#include <cuda/atomic>

__device__ uint permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

__device__ uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {
  return (workgroup_id + 1 + local_id % (testing_workgroups - 1)) % testing_workgroups;
}

__device__ void spin(cuda::atomic<uint, cuda::thread_scope_device>* barrier, uint limit) {
  int i = 0;
  uint val = barrier->fetch_add(1, cuda::memory_order_relaxed);
  while (i < 1024 && val < limit) {
    val = barrier->load(cuda::memory_order_relaxed);
    i++;
  }
}

__device__ void do_stress(uint* scratchpad, uint* scratch_locations, uint iterations, uint pattern) {
  for (uint i = 0; i < iterations; i++) {
    if (pattern == 0) {
      scratchpad[scratch_locations[blockIdx.x]] = i;
      scratchpad[scratch_locations[blockIdx.x]] = i + 1;
    }
    else if (pattern == 1) {
      scratchpad[scratch_locations[blockIdx.x]] = i;
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) {
        break;
      }
    }
    else if (pattern == 2) {
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) {
        break;
      }
      scratchpad[scratch_locations[blockIdx.x]] = i;
    }
    else if (pattern == 3) {
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) {
        break;
      }
      uint tmp2 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp2 > 100) {
        break;
      }
    }
  }
}

__global__ void litmus_test(
  cuda::atomic<uint, cuda::thread_scope_device>* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params) {

  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

    // defined for different distributions of threads across threadblocks
    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);
    uint id_1 = new_workgroup * blockDim.x + threadIdx.x;
    uint id_2 = new_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x);

    // defined for all three thread two memory locations tests
    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2;
    uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 2;
    uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    // pre-stress
    if (kernel_params->pre_stress) {
      do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern);
    }
    if (kernel_params->barrier) {
      spin(barrier, blockDim.x * kernel_params->testing_workgroups);
    }


    if (id_0 != id_1 && id_1 != id_2 && id_0 != id_2) {

      // Thread 0
      test_locations[x_0].store(1, cuda::memory_order_relaxed);

      // Thread 1
      uint r0 = test_locations[x_1].load(cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_block);
      test_locations[y_1].store(1, cuda::memory_order_relaxed);

      // Thread 2
      uint r1 = test_locations[y_2].load(cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_block);
      uint r2 = test_locations[x_2].load(cuda::memory_order_relaxed);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_2].r1 = r1;
      read_results[id_2].r2 = r2;
    }
  } else if (kernel_params->mem_stress) {
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->pre_stress_iterations);
  }
}

__global__ void check_results(
  cuda::atomic<uint, cuda::thread_scope_device>* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params) {
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

