// litmus.cuh
#ifndef LITMUS_CUH
#define LITMUS_CUH

#include <cuda_runtime.h>
#include <cuda/atomic>

#ifdef VOLATILE 
typedef volatile uint d_uint_type;
#elif defined(RMW)
typedef uint d_atomic_uint;
#endif

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
  d_uint_type* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances);

__global__ void check_results(
  d_uint_type* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak);

int host_check_results(TestResults* results, bool print);

#endif
