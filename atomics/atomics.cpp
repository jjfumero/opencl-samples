/**
 * OpenCL Atomics example
 *
 * Author: Juan Fumero <juan.fumero@manchester.ac.uk>
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

int elements = 512;

// Variables
size_t datasize;
int *A;	
int *counter;	


string platformName;
cl_uint numPlatforms;
cl_platform_id *platforms;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueue;
cl_kernel kernel;
cl_program program;
char *source;

cl_mem d_A; 
cl_mem ddA; 
cl_mem d_Counter; 
cl_mem ddCounter; 

cl_event kernelEvent;
cl_event writeEvent1;
cl_event writeEvent2;
cl_event readEvent1;

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
	if (status != CL_SUCCESS || commandQueue == NULL) {
		cout << "Error in create command" << endl;
		return -1;
	}

	const char *sourceFile = "kernel.cl";
	source = readsource(sourceFile);
	program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &status);
	if (status != CL_SUCCESS) {
		cout << "Error creating the program" << endl;
	}

	const char* flags = "-cl-std=CL2.0";
	cl_int buildErr = clBuildProgram(program, numDevices, devices, flags, NULL, NULL);
	if (buildErr != CL_SUCCESS) {
		cout << "Error building the program" << endl;
		abort();
	}

	kernel = clCreateKernel(program, "atomics", &status);
	if (status != CL_SUCCESS) {
		cout << "Error creating the kernel" << endl;
		abort();
	}
}

int hostDataInitialization(int elements) {
	datasize = sizeof(int)*elements;

	ddA = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, datasize, NULL, NULL);
	ddCounter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(int), NULL, NULL);
	A = (int*) clEnqueueMapBuffer(commandQueue, ddA, CL_TRUE, CL_MAP_READ, 0, datasize, 0, NULL, NULL, NULL);
	counter = (int*) clEnqueueMapBuffer(commandQueue, ddCounter, CL_TRUE, CL_MAP_READ, 0, sizeof(int), 0, NULL, NULL, NULL);
	#pragma omp parallel for 
	for (int i = 0; i < elements; i++) {
		A[i] = 0;
	}
}

int allocateBuffersOnGPU() {
	cl_int status;
 	d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, elements * sizeof(int), NULL, NULL);
	d_Counter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
}

int writeBuffer() {
	clEnqueueWriteBuffer(commandQueue, d_A, CL_FALSE, 0, elements * sizeof(int), A, 0, NULL, &writeEvent1);
	clEnqueueWriteBuffer(commandQueue, d_Counter, CL_FALSE, 0, sizeof(int), counter, 0, NULL, &writeEvent2);
}

int runKernel() {
	cl_int status;
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	status  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_Counter);
	
	// launch kernel 
	size_t globalWorkSize[1];
	globalWorkSize[0] = elements;

	size_t localWorkSize[1] = {16};

	clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvent);

	kernelTime = getTime(kernelEvent);

	// Read result back from device to host	
	cl_event waitEventsRead[] = {kernelEvent};
	clEnqueueReadBuffer(commandQueue, d_A, CL_TRUE, 0,  sizeof(int)*elements, A, 0, NULL , &readEvent1);
}

void freeMemory() {
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(d_A);
	cl_int status = clReleaseContext(context);
	status = clReleaseContext(context);
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
		elements = atoi(argv[1]);
	}

	cout << "OpenCL Atomics " << endl;
	cout << "Size = " << elements << endl;

	vector<long> kernelTimers;
	vector<long> writeTimers;
	vector<long> readTimers;
	vector<double> totalTime;

	openclInitialization();
	hostDataInitialization(elements);
	allocateBuffersOnGPU();

	for (int i = 0; i < 1; i++) {

		kernelTime = 0;
		writeTime = 0;
		readTime = 0;

	    auto start_time = chrono::high_resolution_clock::now();
		writeBuffer();
		runKernel();
	  	auto end_time = chrono::high_resolution_clock::now();

		writeTime = getTime(writeEvent1);
		kernelTime = getTime(kernelEvent);
		readTime = getTime(readEvent1);

		kernelTimers.push_back(kernelTime);
		writeTimers.push_back(writeTime);
		readTimers.push_back(readTime);
	
		double total = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
  		totalTime.push_back(total);	

		if (CHECK_RESULT) {
			for (int i = 0; i < elements; i++) {
				cout << "Result: " << i <<": " << A[i] << endl;
			}
		}

		// Print info ocl timers
		cout << "Iteration: " << i << endl;
		cout << "Write    : " <<  writeTime  << endl;
		cout << "X        : " <<  kernelTime  << endl;
		cout << "Reading  : " <<  readTime  << endl;
		cout << "C++ total: " << total << endl;
		cout << "\n";
	}
	
	freeMemory();

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

