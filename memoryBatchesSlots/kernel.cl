
kernel void compute(global float *input) {
	int idx = get_global_id(0);
	input[idx] = 100;
}
