__kernel void mxm(__global float *a,  __global float *b, __global float *c, const int n) {

	uint idx = get_global_id(0);
	uint jdx = get_global_id(1);

	float sum = 0.0;
	for (int k = 0; k < n; k++) {
		sum += a[idx * n + k] * b[k * n + jdx];
	}

	c[idx * n + jdx]  =  sum;
}


__kernel void mxmLI(__global float *a,  __global float *b, __global float *c, int n) {

	uint idx = get_global_id(1);
	uint jdx = get_global_id(0);

	float sum = 0.0f;
	for (int k = 0; k < n; k++) {
		sum += a[idx * n + k] * b[k * n + jdx];
	}

	c[idx * n + jdx]  =  sum;
}

__kernel void mxmLIfma(__global float *a,  __global float *b, __global float *c, int n) {

	uint idx = get_global_id(1);
	uint jdx = get_global_id(0);

	float sum = 0.0f;
	for (int k = 0; k < n; k++) {
	    float op1 = a[idx * n + k];
	    float op2 = b[k * n + jdx];
	    sum = fma(op1, op2, sum);
	}

	c[idx * n + jdx]  =  sum;
}

__kernel void mxmLIfmaUnroll(__global float *a,  __global float *b, __global float *c, int n) {

	uint idx = get_global_id(1);
	uint jdx = get_global_id(0);

	float sum = 0.0;

	#pragma unroll 4
	for (int k = 0; k < n; k++) {
	    float op1 = a[idx * n + k];
	    float op2 = b[k * n + jdx];
	    sum = fma(op1, op2, sum);
	}

	c[idx * n + jdx]  =  sum;
}