## Matrix Multiplication in OpenCL

This program shows different kernels for the Matrix Multiplication with different optimizations.

### How to compile?

```bash
make
```

### How to run? 

```bash
./mxm
```

### Options:

```bash
./mxm -h
Options: 
	 -p <number>       Select an OpenCL Platform Number
	 -s <size>         Select input matrix size
	 -k <kernel name>  Input Kernel <mxm | mxmLI | mxmLIfma | mxmLIfmaUnroll>
	 -w <nThreads>     Select local work group size <nThreads x nThreads>. If not selected, then it sets to NULL
	 -f                Apply optimizations in the compiler flags when building the kernel (-cl-mad-enable -cl-fast-relaxed-math -w)
	 -c                Check results
	 -h                Show this help
```


#### Examples:

Run on platform `1` with size `1024x1024`, checking results `on` and kernel `mxmLI`:

```bash
./mxm -c -p 1 -s 1024 -k mxmLI
```

Select mxmLI with a block of threads of 16x16

```bash
./mxm -s 1024 -k mxmLI -w 16
```


### Benchmarking 

```bash
./benchmarks.sh <platformNumber>

## Then check the generated log* directory

# Filter Total C++ timers for each configuration 

./filter.sh log_2024-11-14_10_52_34/mxm 
```
