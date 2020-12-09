
//__global atomic_int i_1 = ATOMIC_VAR_INIT(0);
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
// Atomics example in OpenCL
//__kernel void atomics(__global int *input, global int *counter) {
//__kernel void atomics(__global int *input) {
//   int idx = get_global_id(0);
//   input[idx] = atomic_fetch_add_explicit(&i_1, 2,  memory_order_relaxed);
//}

// Using an extra parameter per atomic
__kernel void atomics2(__global int *input, global int *counter) {
   int idx = get_global_id(0);
   int value = atomic_add(&counter[0], 2);
   input[idx] = value;
}

// Using a sync variable declared in global memory
__kernel void atomics3(__global int *input, global int *counter) {
   int idx = get_global_id(0);
   input[idx] = atomic_fetch_add_explicit(&i_1, 2,  memory_order_relaxed);
}
