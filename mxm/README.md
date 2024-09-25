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
	 -s <size>         Select input size
	 -k <kernel name>  Input Kernel <mxm|mxmLI|mxmLIfma>
	 -c                Check results
	 -h                Show this help
```


#### Examples:


Run on platform 1 with size `1024x1024`, checking results and kernel `mxmLI`:

```bash
./mxm -c -p 1 -s 1024 -k mxmLI
```
