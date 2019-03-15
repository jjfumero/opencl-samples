# First approach to batch processing with OpenCL (When data can fit into the GPU)

The first approach is as follows:

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