# First approach to batch processing with OpenCL (When data can fit into the GPU)

The first approach is as follows: it splits the iterations space en equal parts for `COPY_IN` and `COPY_OUT`, but there is only one launch and one command queue. 

1. Write Buffer 1 (half space) into command queue Q.
1. Write Buffer 2 (half space) into command queue Q.
1. Launch Kernel
1. Read Buffer 1 (half space) from command queue Q.
1. Read Buffer 2 (half space) from command queue Q.


### Some experiments

```bash
# Run chunks of 1GB
 ./test01 500000000  # running 2GB into 2 chunks of 1GB
```


```bash
# In my local machine/laptop GTX 1050
$./test00 800000000 
 ERROR
```

But if I run in 2 chunks
```bash
# In my local machine/laptop GTX 1050
$./test01 800000000 
Correct
```