/**
 * Test for multiple execution slots for allowing batch processing. 
 * 
 * This strategy works when the host buffer size is bigger than the GPU global memory. 
 * 
 * Date: 15/03/2019
 * 
 * 2**31 = 2147483648 elements * 4 bytes (float ops) = 8GB of data
 * 
 * 512MB ~ 536870912 floats elements. 
 * 
 */

#include <iostream>
#include <string>
#include <omp.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "../common/readSource.h"
using namespace std;

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

const bool CHECK_RESULT = true;

#ifdef __APPLE__
	#include <OpenCL/cl.h>
#else
	#include <CL/cl.h>
#endif

const int PLATFORM_ID = 0;
const bool DEBUG = true;

const bool USE_PINNED_MEMORY = false;

long elements          = 4294967296;
const long floatElements512MB = 134217728;
int iterations = elements/floatElements512MB;

size_t datasize;
float *A;	

string platformName;
cl_uint numPlatforms;
cl_platform_id *platforms;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueue;
cl_command_queue commandQueue2;
cl_kernel kernel;
cl_program program;
char *source;

cl_mem d_A; 

cl_event kernelEvent1;
cl_event kernelEvent2;
cl_event writeEvent1;
cl_event writeEvent2;
cl_event writeEvent3;
cl_event writeEvent4;
cl_event readEvent1;
cl_event readEvent2;

long kernelTime;
long writeTime;
long readTime;

long getTime(cl_event event) {
    clWaitForEvents(1, &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    return (time_end - time_start);
}

int openclInitialization() {

	cl_int status;	
	cl_uint numPlatforms = 0;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	if (numPlatforms == 0) {
		cout << "No platform detected" << endl;
		return -1;
	}

	platforms = (cl_platform_id*) malloc(numPlatforms*sizeof(cl_platform_id));
	if (platforms == NULL) {
		cout << "malloc platform_id failed" << endl;
		return -1;
	}
	
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		cout << "clGetPlatformIDs failed" << endl;
		return -1;
	}	

	cout << numPlatforms <<  " has been detected" << endl;
	for (int i = 0; i < numPlatforms; i++) {
		char buf[10000];
		cout << "Platform: " << i << endl;
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
		if (i == PLATFORM_ID) {
			platformName += buf;
		}
		cout << "\tVendor: " << buf << endl;
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
	}

	
	cl_uint numDevices = 0;

	cl_platform_id platform = platforms[PLATFORM_ID];
	std::cout << "Using platform: " << PLATFORM_ID << " --> " << platformName << std::endl;

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	
	if (status != CL_SUCCESS) {
		cout << "[WARNING] Using CPU, no GPU available" << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	} else {
		devices = (cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}
	
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if (context == NULL) {
		cout << "Context is not NULL" << endl;
	} 
	
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);	
	commandQueue2 = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);	

	if (status != CL_SUCCESS || commandQueue == NULL) {
		cout << "Error in create command" << endl;
		return -1;
	}

	const char *sourceFile = "kernel.cl";
	source = readsource(sourceFile);
	program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &status);

	cl_int buildErr;
	buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "compute", &status);
	if (status != CL_SUCCESS) {
		cout << "Error in KERNEL compilation" << endl;
	}
}

// Alloc a big chunk of host memory > 8GB
void hostDataInitialization(long elements) {
	datasize = sizeof(float)*elements;
	cl_int status;

	if (!USE_PINNED_MEMORY) {
		cout << "Normal malloc: " << elements << endl;
		A = (float*) malloc(datasize);
	} else {
		cl_mem ddA = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, datasize, NULL, &status);
		if (status != CL_SUCCESS) {
			cout << "Error allocating buffer on the HOST  (A)\n";
		}
		A = (float*) clEnqueueMapBuffer(commandQueue, ddA, CL_TRUE, CL_MAP_WRITE, 0, datasize, 0, NULL, NULL, NULL);
	}

	for (long i = 0; i < elements; i++) {
		A[i] = i;
	}
}

// On the GPU we alloc max of 512MB
int allocateBuffersOnGPU(long size) {
	cl_int status;
 	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &status);
	if (status != CL_SUCCESS) {
		cout << "Error allocating buffer on the GPU (A)\n";
	}
}

// Playing with offsets 
int writeBuffer1(long sizeToCopy, long offset, long numEvents, cl_event eventsToWait[], cl_event *eventToWrite) {
	if (DEBUG) { 
		cout << "[DEBUG] Size TO COPY: " << (sizeToCopy / sizeof(float)) << " Offset: " << offset<< endl;
	}
	// The key part is to start always in the index 0 for the device buffer
	cl_int status = clEnqueueWriteBuffer(commandQueue, d_A, CL_FALSE, 0, sizeToCopy, &A[offset], numEvents, eventsToWait, eventToWrite);
	if (DEBUG) { 
		cout << "\t[DEBUG] ERROR CODE: " << status << endl;
	}
	if (status != CL_SUCCESS) {
		cout << "Error copying data A\n";
	}
	return 0;
}

void runKernel(long threadsToRun, size_t offset) {
	cl_int status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	if (status != CL_SUCCESS) {
			cout << "Error Kernel Parameters\n";
	}

	// lauch kernel 
	size_t globalWorkSize[1];
	globalWorkSize[0] = threadsToRun;

	//size_t offsets[] = {offset};

	status = clEnqueueNDRangeKernel(commandQueue,  kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &kernelEvent1);

	if (status != CL_SUCCESS) {
		cout << "Error in Launch\n";
		if (status == CL_INVALID_WORK_DIMENSION) {
			cout << "[KERNEL] Invalid work dimensions\n";	
		} else if (status == CL_INVALID_GLOBAL_WORK_SIZE) {
			cout << "[KERNEL] Invalid work size\n";	
		} else if (status == CL_OUT_OF_RESOURCES) {
			cout << "[KERNEL] Out of resources \n";	
		} else if (status == CL_OUT_OF_HOST_MEMORY) {
			cout << "[KERNEL] Out of HOST Memory \n";	
		}
	}
}

void readBuffer(long sizeToCopyBack, long offset) {
	// Read result back from device to host	
	cl_int status = clEnqueueReadBuffer(commandQueue, d_A, CL_FALSE, 0, sizeToCopyBack, &A[offset], 0, NULL, &readEvent1);
	if (status != CL_SUCCESS) {
		cout << "Error Reading Buffer. Error code: " << status << endl;
		if (status == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
			cout << "[ERROR] CL_MEM_OBJECT_ALLOCATION_FAILURE " << endl;
		}
	}
}

void freeMemory() {
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(d_A);
	clReleaseContext(context);
	clReleaseContext(context);
	free(source);
	free(platforms);
	free(devices);
}

double median(vector<long> data) {
	if(data.empty()) {
		return 0;
	}
	else {
    	sort(data.begin(), data.end());
	    if(data.size() % 2 == 0) {
			return (data[data.size()/2 - 1] + data[data.size()/2]) / 2;
		}
    	else {
			return double(data[data.size()/2]);
		}
	}
}

double median(vector<double> data) {
	if(data.empty()) {
		return 0;
	}
	else {
    	sort(data.begin(), data.end());
	    if(data.size() % 2 == 0) {
			return (data[data.size()/2 - 1] + data[data.size()/2]) / 2;
		}
    	else {
			return double(data[data.size()/2]);
		}
	}
}

int main(int argc, char **argv) {

	if (argc > 1) {
		elements = atol(argv[1]);
		iterations = elements/floatElements512MB;
	}

	cout << "OpenCL Big Copy Test " << endl;
	cout << "Size = " << elements << endl;

	vector<long> kernelTimers;
	vector<long> writeTimers;
	vector<long> readTimers;
	vector<double> totalTime;

	openclInitialization();
	hostDataInitialization(elements);
	allocateBuffersOnGPU(floatElements512MB * sizeof(float));

	// 16 slots of 512 MB  = 8GB
	cout << "Number of Iterations: " << iterations << endl;
	for (int i = 0; i < iterations; i++) {

		kernelTime = 0;
		writeTime = 0;
		readTime = 0;

	  auto start_time = chrono::high_resolution_clock::now();
		
		long offset = floatElements512MB * i;
		long batchSize =  floatElements512MB * sizeof(4);

		writeBuffer1(batchSize, offset, 0, NULL, &writeEvent1);
		runKernel(floatElements512MB, 0);
		readBuffer(batchSize, offset);

	  auto end_time = chrono::high_resolution_clock::now();

		// Writing timers
		writeTime = getTime(writeEvent1);
		writeTime += getTime(writeEvent2);

		// Computing timers
		kernelTime = getTime(kernelEvent1);

		// Reading timers
		readTime = getTime(readEvent1);
		readTime += getTime(readEvent2);

		kernelTimers.push_back(kernelTime);
		writeTimers.push_back(writeTime);
		readTimers.push_back(readTime);
	
		double total = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
  	totalTime.push_back(total);	

		// Print info ocl timers
		cout << "Iteration: " << i << endl;
		cout << "Write    : " <<  writeTime  << endl;
		cout << "X        : " <<  kernelTime  << endl;
		cout << "Reading  : " <<  readTime  << endl;
		cout << "C++ total: " << total << endl;
		cout << "\n";
	}

	clFlush(commandQueue);
	
	freeMemory();

	if (CHECK_RESULT) {
			bool valid = true;
			for (long i = 0; i < elements; i++) {
				if( i != A[i]) {
					cout << "Expected: " << i  << "  but found: " << (A[i]) << " in index: " << i << endl;
					valid = false;
					break;
				}
			}
			if (valid) {
				cout << "Result is correct" << endl;
			} else {
				cout << "Result is not correct" << endl;
			}
		}

	// Compute median
	double medianKernel = median(kernelTimers);
	double medianWrite = median(writeTimers);
	double medianRead = median(readTimers);
	double medianTotalTime = median(totalTime);
	
	cout << "Median KernelTime: " << medianKernel << " (ns)" << endl;
	cout << "Median CopyInTime: " << medianWrite << " (ns)" << endl;
	cout << "Median CopyOutTime: " << medianRead << " (ns)" << endl;
	cout << "Median TotalTime: " << medianTotalTime << " (ns)" << endl;

	return 0;	
}
