#include <iostream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../common/readSource.h"
#include <unistd.h>
using namespace std;

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

const bool CHECK_RESULT = false;

#ifdef __APPLE__
	#include <OpenCL/cl.h>
#else
	#include <CL/cl.h>
#endif

int PLATFORM_ID = 0;
int elements = 1024;

// Variables
size_t datasize;
float *A;	
float *B;
float *C;

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
cl_mem d_B;
cl_mem d_C;

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

	cl_int buildErr;
	buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "mxm", &status);
	if (status != CL_SUCCESS) {
		std::cout << "clCreateKernel error" << std::endl;
	}
}

int hostDataInitialization(int elements) {
	datasize = sizeof(float)*elements * elements;

	A = (float*) malloc(datasize);
	B = (float*) malloc(datasize);
	C = (float*) malloc(datasize);

	for (int i = 0; i < elements; i++) {
		for (int j = 0; j < elements; j++) {
		    A[i * elements + j] = 2.0f;
		    B[i * elements + j] = 2.0f;
		}
	}
}

int allocateBuffersOnGPU() {
	cl_int status;
 	d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);
	d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);
 	d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, NULL);
}

int writeBuffer() {
	clEnqueueWriteBuffer(commandQueue, d_A, CL_TRUE, 0, datasize, A, 0, NULL, &writeEvent1);
	clEnqueueWriteBuffer(commandQueue, d_B, CL_TRUE, 0, datasize, B, 0, NULL, &writeEvent2);
}

int runKernel() {
	cl_int status;
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_int), &elements);
	
	// launch kernel 
	size_t globalWorkSize[2];
	globalWorkSize[0] = elements;
	globalWorkSize[1] = elements;
	cl_event waitEventsKernel[] = {writeEvent1, writeEvent2};
	clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 2, waitEventsKernel, &kernelEvent);

	kernelTime = getTime(kernelEvent);

	// Read result back from device to host	
	cl_event waitEventsRead[] = {kernelEvent};
	clEnqueueReadBuffer(commandQueue, d_C, CL_TRUE, 0,  datasize, C, 1, waitEventsKernel, &readEvent1);
}

void freeMemory() {
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
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

void printHelp() {
	cout << "Options: \n";
	cout << "\t -p <number>   Select an OpenCL Platform Number\n"; 
	cout << "\t -s <size>     Select input size\n"; 
}

void processCommandLineOptions(int argc, char **argv) {
	int option;
	bool doHelp = false;
	while ((option = getopt(argc, argv, ":p:s:h")) != -1) {
        switch (option) {
			case 's':
				elements = atoi(optarg);
				break;
			case 'p':
				PLATFORM_ID = atoi(optarg);
				break;
			case 'h':
				doHelp = true;
				break;
			default:
				cout << "Error" << endl;
				break;
		}
	}
	if (doHelp) {
		printHelp();
		exit(0);
	}
}


int main(int argc, char **argv) {

	processCommandLineOptions(argc, argv);

	cout << "OpenCL MxM " << endl;
	cout << "Size = " << elements << "x" << elements << endl;

	vector<long> kernelTimers;
	vector<long> writeTimers;
	vector<long> readTimers;
	vector<double> totalTime;

	openclInitialization();
	hostDataInitialization(elements);
	allocateBuffersOnGPU();

	for (int i = 0; i < 11; i++) {

		kernelTime = 0;
		writeTime = 0;
		readTime = 0;

	    auto start_time = chrono::high_resolution_clock::now();
		writeBuffer();
		runKernel();
	  	auto end_time = chrono::high_resolution_clock::now();

		writeTime = getTime(writeEvent1);
		writeTime += getTime(writeEvent2);
		kernelTime = getTime(kernelEvent);
		readTime = getTime(readEvent1);

		kernelTimers.push_back(kernelTime);
		writeTimers.push_back(writeTime);
		readTimers.push_back(readTime);
	
		double total = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
  		totalTime.push_back(total);	

		float* r = (float*)malloc(datasize);

		if (CHECK_RESULT) {
			for (int i = 0; i < elements; i++) {
				for (int j = 0; j < elements; j++) {
					r[i * elements + j] = 0;
					for (int k = 0; k < elements; k++) {
		    			r[i * elements + j] += A[i * elements + k] * B[k * elements + j];
					}
				}
			}

			bool valid = true;
			for (int i = 0; i < elements; i++) {
				for (int j = 0; j < elements; j++) {
					if(abs(r[i * elements + j] - C[i * elements + j]) > 0.1) {
						valid = false;
						break;
					}
				}
			}
			if (valid) {
				cout << "Result is correct" << endl;
			} else {
				cout << "Result is not correct" << endl;
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

