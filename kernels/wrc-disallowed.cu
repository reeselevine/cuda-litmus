#include <iostream>
#include <set>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cuda/atomic>

/** Kernel structs and functions */

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
  uint x;
  uint y;
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


/** Test kernel. */

__global__ void litmus_test(
  cuda::atomic<uint, cuda::thread_scope_device>* test_locations,
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
    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);
    uint id_1 = new_workgroup * blockDim.x + threadIdx.x;
    uint id_2 = new_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x);

    // defined for all three thread two memory locations tests
    uint x_0 =  id_0 * kernel_params->mem_stride * 2;
    uint x_1 = id_1 * kernel_params->mem_stride * 2;
    uint y_1 = permute_id(id_1, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_2 = id_2 * kernel_params->mem_stride * 2;
    uint y_2 = permute_id(id_2, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    // Save threads and memory locations involved in a test instance
    uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
    test_instances[id_0].t0 = t_id;
    test_instances[id_1].t1 = t_id;
    test_instances[id_2].t2 = t_id;
    test_instances[id_1].x = x_1;
    test_instances[id_1].y = y_1;

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

/** Result aggregation kernel. */

__global__ void check_results(
  cuda::atomic<uint, cuda::thread_scope_device>* test_locations,
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

/** Device structs and functions. */

typedef struct {
  int numMemLocations;
  int permuteLocation;
} TestParams;

typedef struct {
  int testIterations;
  int testingWorkgroups;
  int maxWorkgroups;
  int workgroupSize;
  int shufflePct;
  int barrierPct;
  int stressLineSize;
  int stressTargetLines;
  int scratchMemorySize;
  int memStride;
  int memStressPct;
  int memStressIterations;
  int memStressPattern;
  int preStressPct;
  int preStressIterations;
  int preStressPattern;
  int stressAssignmentStrategy;
  int permuteThread;
} StressParams;

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

int parseTestParamsFile(const char* filename, TestParams* config) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return -1;
  }

  char line[256];
  while (fgets(line, sizeof(line), file)) {
    char key[64];
    int value;
    if (sscanf(line, "%63[^=]=%d", key, &value) == 2) {
      if (strcmp(key, "numMemLocations") == 0) config->numMemLocations = value;
      else if (strcmp(key, "permuteLocation") == 0) config->permuteLocation = value;
    }
  }

  fclose(file);
  return 0;
}

int parseStressParamsFile(const char* filename, StressParams* config) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return -1;
  }

  char line[256];
  while (fgets(line, sizeof(line), file)) {
    char key[64];
    int value;
    if (sscanf(line, "%63[^=]=%d", key, &value) == 2) {
      if (strcmp(key, "testIterations") == 0) config->testIterations = value;
      else if (strcmp(key, "testingWorkgroups") == 0) config->testingWorkgroups = value;
      else if (strcmp(key, "maxWorkgroups") == 0) config->maxWorkgroups = value;
      else if (strcmp(key, "workgroupSize") == 0) config->workgroupSize = value;
      else if (strcmp(key, "shufflePct") == 0) config->shufflePct = value;
      else if (strcmp(key, "barrierPct") == 0) config->barrierPct = value;
      else if (strcmp(key, "stressLineSize") == 0) config->stressLineSize = value;
      else if (strcmp(key, "stressTargetLines") == 0) config->stressTargetLines = value;
      else if (strcmp(key, "scratchMemorySize") == 0) config->scratchMemorySize = value;
      else if (strcmp(key, "memStride") == 0) config->memStride = value;
      else if (strcmp(key, "memStressPct") == 0) config->memStressPct = value;
      else if (strcmp(key, "memStressIterations") == 0) config->memStressIterations = value;
      else if (strcmp(key, "memStressPattern") == 0) config->memStressPattern = value;
      else if (strcmp(key, "preStressPct") == 0) config->preStressPct = value;
      else if (strcmp(key, "preStressIterations") == 0) config->preStressIterations = value;
      else if (strcmp(key, "preStressPattern") == 0) config->preStressPattern = value;
      else if (strcmp(key, "stressAssignmentStrategy") == 0) config->stressAssignmentStrategy = value;
      else if (strcmp(key, "permuteThread") == 0) config->permuteThread = value;
    }
  }

  fclose(file);
  return 0;
}

/** Returns a value between the min and max (inclusive). */
int setBetween(int min, int max) {
  if (min == max) {
    return min;
  }
  else {
    int size = rand() % (max - min + 1);
    return min + size;
  }
}

bool percentageCheck(int percentage) {
  return rand() % 100 < percentage;
}

/** Assigns shuffled workgroup ids, using the shufflePct to determine whether the ids should be shuffled this iteration. */
void setShuffledWorkgroups(uint* h_shuffledWorkgroups, int numWorkgroups, int shufflePct) {
  for (int i = 0; i < numWorkgroups; i++) {
    h_shuffledWorkgroups[i] = i;
  }
  if (percentageCheck(shufflePct)) {
    for (int i = numWorkgroups - 1; i > 0; i--) {
      int swap = rand() % (i + 1);
      int temp = h_shuffledWorkgroups[i];
      h_shuffledWorkgroups[i] = h_shuffledWorkgroups[swap];
      h_shuffledWorkgroups[swap] = temp;
    }
  }
}

/** Sets the stress regions and the location in each region to be stressed. Uses the stress assignment strategy to assign
  * workgroups to specific stress locations. Assignment strategy 0 corresponds to a "round-robin" assignment where consecutive
  * threads access separate scratch locations, while assignment strategy 1 corresponds to a "chunking" assignment where a group
  * of consecutive threads access the same location.
  */
void setScratchLocations(uint* h_locations, int numWorkgroups, StressParams params) {
  std::set<int> usedRegions;
  int numRegions = params.scratchMemorySize / params.stressLineSize;
  for (int i = 0; i < params.stressTargetLines; i++) {
    int region = rand() % numRegions;
    while (usedRegions.count(region))
      region = rand() % numRegions;
    int locInRegion = rand() % params.stressLineSize;
    switch (params.stressAssignmentStrategy) {
    case 0:
      for (int j = i; j < numWorkgroups; j += params.stressTargetLines) {
        h_locations[j] = (region * params.stressLineSize) + locInRegion;
      }
      break;
    case 1:
      int workgroupsPerLocation = numWorkgroups / params.stressTargetLines;
      for (int j = 0; j < workgroupsPerLocation; j++) {
        h_locations[i * workgroupsPerLocation + j] = (region * params.stressLineSize) + locInRegion;
      }
      if (i == params.stressTargetLines - 1 && numWorkgroups % params.stressTargetLines != 0) {
        for (int j = 0; j < numWorkgroups % params.stressTargetLines; j++) {
          h_locations[numWorkgroups - j - 1] = (region * params.stressLineSize) + locInRegion;
        }
      }
      break;
    }
  }
}

/** These parameters vary per iteration, based on a given percentage. */
void setDynamicKernelParams(KernelParams* h_kernelParams, StressParams params) {
  h_kernelParams->barrier = percentageCheck(params.barrierPct);
  h_kernelParams->mem_stress = percentageCheck(params.memStressPct);
  h_kernelParams->pre_stress = percentageCheck(params.preStressPct);
}

/** These parameters are static for all iterations of the test. */
void setStaticKernelParams(KernelParams* h_kernelParams, StressParams stressParams, TestParams testParams) {
  h_kernelParams->mem_stress_iterations = stressParams.memStressIterations;
  h_kernelParams->mem_stress_pattern = stressParams.memStressPattern;
  h_kernelParams->pre_stress_iterations = stressParams.preStressIterations;
  h_kernelParams->pre_stress_pattern = stressParams.preStressPattern;
  h_kernelParams->permute_thread = stressParams.permuteThread;
  h_kernelParams->permute_location = testParams.permuteLocation;
  h_kernelParams->testing_workgroups = stressParams.testingWorkgroups;
  h_kernelParams->mem_stride = stressParams.memStride;
  h_kernelParams->mem_offset = stressParams.memStride;
}

int total_behaviors(TestResults * results) {
  return results->res0 + results->res1 + results->res2 + results->res3 + 
  results->res4 + results->res5 + results->res6 + results->res7 + 
  results->res8 + results->res9 + results->res10 + results->res11 + 
  results->res12 + results->res13 + results->res14 + results->res15 + 
  results->weak + results->other;
}


void run(StressParams stressParams, TestParams testParams, bool print_results) {
  int testingThreads = stressParams.workgroupSize * stressParams.testingWorkgroups;

  int testLocSize = testingThreads * testParams.numMemLocations * stressParams.memStride * sizeof(uint);
  cuda::atomic<uint, cuda::thread_scope_device>* testLocations;
  cudaMalloc(&testLocations, testLocSize);

  int readResultsSize = sizeof(ReadResults) * testingThreads;
  ReadResults* readResults;
  cudaMalloc(&readResults, readResultsSize);

  TestResults* h_testResults = (TestResults*)malloc(sizeof(TestResults));
  TestResults* d_testResults;
  cudaMalloc(&d_testResults, sizeof(TestResults));

  int shuffledWorkgroupsSize = stressParams.maxWorkgroups * sizeof(uint);
  uint* h_shuffledWorkgroups = (uint*)malloc(shuffledWorkgroupsSize);
  uint* d_shuffledWorkgroups;
  cudaMalloc(&d_shuffledWorkgroups, shuffledWorkgroupsSize);

  int barrierSize = sizeof(uint);
  cuda::atomic<uint, cuda::thread_scope_device>* barrier;
  cudaMalloc(&barrier, barrierSize);

  int scratchpadSize = stressParams.scratchMemorySize * sizeof(uint);
  uint* scratchpad;
  cudaMalloc(&scratchpad, scratchpadSize);

  int scratchLocationsSize = stressParams.maxWorkgroups * sizeof(uint);
  uint* h_scratchLocations = (uint*)malloc(scratchLocationsSize);
  uint* d_scratchLocations;
  cudaMalloc(&d_scratchLocations, scratchLocationsSize);

  KernelParams* h_kernelParams = (KernelParams*)malloc(sizeof(KernelParams));
  KernelParams* d_kernelParams;
  cudaMalloc(&d_kernelParams, sizeof(KernelParams));
  setStaticKernelParams(h_kernelParams, stressParams, testParams);

  int testInstancesSize = sizeof(TestInstance) * testingThreads;
  TestInstance* h_testInstances = (TestInstance*)malloc(testInstancesSize);
  TestInstance* d_testInstances;
  cudaMalloc(&d_testInstances, testInstancesSize);

  int weakSize = sizeof(bool) * testingThreads;
  bool* h_weak = (bool*)mallo(weakSize);
  bool* d_weak;
  cudaMalloc(&d_weak, weakSize);

  // run iterations
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int weakBehaviors = 0;
  int totalBehaviors = 0;

  for (int i = 0; i < stressParams.testIterations; i++) {
    int numWorkgroups = setBetween(stressParams.testingWorkgroups, stressParams.maxWorkgroups);

    // clear memory
    cudaMemset(testLocations, 0, testLocSize);
    cudaMemset(d_testResults, 0, sizeof(TestResults));
    cudaMemset(readResults, 0, readResultsSize);
    cudaMemset(barrier, 0, barrierSize);
    cudaMemset(scratchpad, 0, scratchpadSize);
    cudaMemset(d_testInstances, 0, testInstancesSize);
    cudaMemset(d_weak, false, weakSize);

    setShuffledWorkgroups(h_shuffledWorkgroups, numWorkgroups, stressParams.shufflePct);
    cudaMemcpy(d_shuffledWorkgroups, h_shuffledWorkgroups, shuffledWorkgroupsSize, cudaMemcpyHostToDevice);
    setScratchLocations(h_scratchLocations, numWorkgroups, stressParams);
    cudaMemcpy(d_scratchLocations, h_scratchLocations, scratchLocationsSize, cudaMemcpyHostToDevice);
    setDynamicKernelParams(h_kernelParams, stressParams);
    cudaMemcpy(d_kernelParams, h_kernelParams, sizeof(KernelParams), cudaMemcpyHostToDevice);

    litmus_test << <numWorkgroups, stressParams.workgroupSize >> > (testLocations, readResults, d_shuffledWorkgroups, barrier, scratchpad, d_scratchLocations, d_kernelParams, d_testInstances);

    check_results << <stressParams.testingWorkgroups, stressParams.workgroupSize >> > (testLocations, readResults, d_testResults, d_kernelParams, d_weak);

    cudaMemcpy(h_testResults, d_testResults, sizeof(TestResults), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_testInstances, d_testInstances, testInstancesSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weak, d_weak, weakSize, cudaMemcpyDeviceToHost);

    if (print_results) {
      std::cout << "Iteration " << i << "\n";
      for (uint i = 0; i < testingThreads; i++) {
        if (h_weak[i]) {
          std:cout << "Weak result " << i << "\n";
          std:cout << "  t0: " << h_testInstances[i].t0;
          std:cout << " t1: " << h_testInstances[i].t1;
          std:cout << " t2: " << h_testInstances[i].t2 << "\n";
          std::cout << "  x: " << h_testInstances[i].x;
          std::cout << " y: " << h_testInstances[i].y << "\n";
        }
      }
    }
    weakBehaviors += host_check_results(h_testResults, print_results);
    totalBehaviors += total_behaviors(h_testResults);
  }

  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
  std::cout << std::fixed << std::setprecision(0) << "Weak behavior rate: " << float(weakBehaviors) / duration.count() << " per second\n";

  std::cout << "Total behaviors: " << totalBehaviors << "\n";
  std::cout << "Number of weak behaviors: " << weakBehaviors << "\n";

  // Free memory
  cudaFree(testLocations);
  cudaFree(readResults);
  cudaFree(d_testResults);
  free(h_testResults);
  cudaFree(d_shuffledWorkgroups);
  free(h_shuffledWorkgroups);
  cudaFree(barrier);
  cudaFree(scratchpad);
  cudaFree(d_scratchLocations);
  free(h_scratchLocations);
  cudaFree(d_kernelParams);
  free(h_kernelParams);
}

int main(int argc, char* argv[]) {
  char* stress_params_file = nullptr;
  char* test_params_file = nullptr;
  bool print_results = false;

  int c;
  while ((c = getopt(argc, argv, "xs:t:")) != -1)
    switch (c)
    {
    case 's':
      stress_params_file = optarg;
      break;
    case 't':
      test_params_file = optarg;
      break;
    case 'x':
      print_results = true;
      break;
    case '?':
      if (optopt == 's' || optopt == 't')
        std::cerr << "Option -" << optopt << "requires an argument\n";
      else
        std::cerr << "Unknown option" << optopt << "\n";
      return 1;
    default:
      abort();
    }

  if (stress_params_file == nullptr) {
    std::cerr << "Stress param file (-s) must be set\n";
    return 1;
  }

  if (test_params_file == nullptr) {
    std::cerr << "Test param file (-t) must be set\n";
    return 1;
  }

  StressParams stressParams;
  if (parseStressParamsFile(stress_params_file, &stressParams) != 0) {
    return 1;
  }

  TestParams testParams;
  if (parseTestParamsFile(test_params_file, &testParams) != 0) {
    return 1;
  }
  run(stressParams, testParams, print_results);
}

