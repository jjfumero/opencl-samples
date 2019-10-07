## Saxpy Apps with Zero copy example for Intel Integrated Graphics

```bash
$ make 
$ ./saxpy <size>
```


### How to enable Zero copy in intel Integrated graphics

The way it works is as follows:

```C
// 1. Create buffer
ddA = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, datasize, NULL, NULL);

// 2. Map the buffer
A = (double*) clEnqueueMapBuffer(commandQueue, ddA, CL_TRUE, CL_MAP_WRITE, 0, datasize, 0, NULL, NULL, NULL);

// 3. Write into the buffer in the host pointer
for (int i = 0; i < elements; i++) {
		A[i] = 2;
		B[i] = 4;
}

// 4. Write into buffer -> USE CL_TRUE
clEnqueueWriteBuffer(commandQueue, ddA, CL_TRUE, 0, elements * sizeof(double), A, 0, NULL, &writeEvent1);

// ... 
status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ddA);
```

**NOTE**: Write Buffer operation has to be blocking


Example of execution:

Platform and driver version using `clinfo`:

```bash
  Name:						 Intel(R) Gen9 HD Graphics NEO
  Vendor:					 Intel(R) Corporation
  Device OpenCL C version:			 OpenCL C 2.0 
  Driver version:				 19.34.13959
  Profile:					 FULL_PROFILE
  Version:					 OpenCL 2.1 NEO 
```


```bash
OpenCL Saxpy 
Size = 33554432
4 has been detected
Platform: 0
	Vendor: NVIDIA Corporation
Platform: 1
	Vendor: Intel(R) Corporation
Platform: 2
	Vendor: Advanced Micro Devices, Inc.
Platform: 3
	Vendor: Intel(R) Corporation
Using platform: 3 --> Intel(R) Corporation 
Result is correct
Iteration: 0
Write    : 1684
X        : 37784000
Reading  : 21370
C++ total: 4.05873e+07

Result is correct
Iteration: 1
Write    : 1395
X        : 38451333
Reading  : 793
C++ total: 4.00289e+07

Result is correct
Iteration: 2
Write    : 1403
X        : 38035000
Reading  : 765
C++ total: 3.92745e+07

Result is correct
Iteration: 3
Write    : 1389
X        : 37978083
Reading  : 736
C++ total: 3.91959e+07

Result is correct
Iteration: 4
Write    : 1382
X        : 38446750
Reading  : 730
C++ total: 3.98169e+07

Result is correct
Iteration: 5
Write    : 1380
X        : 38718083
Reading  : 774
C++ total: 4.03948e+07

Result is correct
Iteration: 6
Write    : 1378
X        : 38017500
Reading  : 753
C++ total: 3.9191e+07

Result is correct
Iteration: 7
Write    : 1389
X        : 38077083
Reading  : 740
C++ total: 3.93058e+07

Result is correct
Iteration: 8
Write    : 1392
X        : 38763666
Reading  : 962
C++ total: 4.04242e+07

Result is correct
Iteration: 9
Write    : 1392
X        : 38766250
Reading  : 786
C++ total: 4.06384e+07

Result is correct
Iteration: 10
Write    : 1385
X        : 38573750
Reading  : 727
C++ total: 4.00318e+07

Median KernelTime: 3.84468e+07 (ns)
Median CopyInTime: 1389 (ns)
Median CopyOutTime: 765 (ns)
Median TotalTime: 4.00289e+07 (ns)
```


### ISSUES:

I think there is a bug in the Intel implementation to get the timers. They do not add up

