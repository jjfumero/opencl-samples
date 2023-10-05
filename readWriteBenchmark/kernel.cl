__kernel void readWritePlain(__global float *a,  __global float *b) {	
	uint idx = get_global_id(0);
	b[idx]  =  a[idx];
}

__kernel void readWrite(__global float *a,  __global float *b) {	
	uint idx = get_global_id(0);
	ulong baseA = a; 
	ulong indexA = baseA + (idx << 2);
	ulong baseB = b;
	ulong indexB = baseB + (idx << 2);
	// load 
	float value = *((__global float *) indexA);
	// store
	*((__global float *) indexB)  = value;
}

__kernel void readWriteWithOffset16(__global float *a,  __global float *b) {	
	uint idx = get_global_id(0);
	ulong baseA = ((ulong) a) + 16L;
	ulong indexA = baseA + (idx << 2);
	ulong baseB = ((ulong) b) + 16L;
	ulong indexB = baseB + (idx << 2);
	// load 
	float value = *((__global float *) indexA);
	// store
	*((__global float *) indexB)  = value;
}

__kernel void readWriteWithOffset20(__global float *a,  __global float *b) {	
	uint idx = get_global_id(0);
	ulong baseA = ((ulong) a) + 20L;
	ulong indexA = baseA + (idx << 2);
	ulong baseB = ((ulong) b) + 20L;
	ulong indexB = baseB + (idx << 2);
	// load 
	float value = *((__global float *) indexA);
	// store
	*((__global float *) indexB)  = value;
}

__kernel void readWriteWithOffset24(__global float *a,  __global float *b) {	
	uint idx = get_global_id(0);
	ulong baseA = ((ulong) a) + 24L;
	ulong indexA = baseA + (idx << 2);
	ulong baseB = ((ulong) b) + 24L;
	ulong indexB = baseB + (idx << 2);
	// load 
	float value = *((__global float *) indexA);
	// store
	*((__global float *) indexB)  = value;
}

__kernel void readWriteWithOffset128(__global float *a,  __global float *b) {	
	uint idx = get_global_id(0);
	ulong baseA = ((ulong) a) + 128L;
	ulong indexA = baseA + (idx << 2);
	ulong baseB = ((ulong) b) + 128L;
	ulong indexB = baseB + (idx << 2);
	// load 
	float value = *((__global float *) indexA);
	// store
	*((__global float *) indexB)  = value;
}
