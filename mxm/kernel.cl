
// OpenCL Kernel for MxM 

__kernel void mxm(__global float *a, 
				    __global float *b, 
				     __global float *c
					int size) {	

	uint idx = get_global_id(0);
	uint jdx = get_global_id(1);

	for (int k = 0; k < size; k++) {
		sum += a[idx * size + k] * b[k * size + jdx];
	}

	c[idx * size + jdx]  =  sum;
}


