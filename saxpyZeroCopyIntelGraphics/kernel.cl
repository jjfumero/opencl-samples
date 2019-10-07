
// Saxpy OpenCL kernel 

__kernel void saxpy(__global double *a, 
				    __global double *b, 
				     __global double *c, 	
					const double alpha, 
					int iNumElements) {
	int idx = get_global_id(0);
	if (idx >= iNumElements) {
		return;
	}
	c[idx]  =  a[idx] * alpha + b[idx];
}


