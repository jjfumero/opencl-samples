__kernel void mxm(__global float *a,  __global float *b, __global float *c, int size) {	

	uint idx = get_global_id(0);
	uint jdx = get_global_id(1);

	float sum = 0.0f;
	for (int k = 0; k < size; k++) {
		sum += a[idx * size + k] * b[k * size + jdx];
	}

	c[idx * size + jdx]  =  sum;
}


__kernel void mxmLI(__global float *a,  __global float *b, __global float *c, int size) {	

	uint idx = get_global_id(1);
	uint jdx = get_global_id(0);

	float sum = 0.0f;
	for (int k = 0; k < size; k++) {
		sum += a[idx * size + k] * b[k * size + jdx];
	}

	c[idx * size + jdx]  =  sum;
}
