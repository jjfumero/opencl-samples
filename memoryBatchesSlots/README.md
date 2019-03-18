# Second approach to batch processing with OpenCL 

This approach assumes that the host data does not fit on the GPU global memory and split the execution in batches. 
Each batch computes:

1. Write  <batch>
1. Execution <batch>
1. Copy out <batch>


It uses the same command queue but host data has the whole array (let's say 16GB). Then, 
the device buffer uses a subsection (e.g., 1GB). Then runtime then performs multiple writes, execution and copy out until execute the whole data. 

