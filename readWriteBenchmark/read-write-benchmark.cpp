#include <iostream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <set>
#include <algorithm>
#include "../common/readSource.h"
#include <unistd.h>
using namespace std;

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

const bool CHECK_RESULT = true;

#ifdef __APPLE__
	#include <OpenCL/cl.h>
#else
	#include <CL/cl.h>
#endif

int PLATFORM_ID = 0;
unsigned long elements = 1024;

// Variables
size_t datasize;
char *A;	
char *B;

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

int VERSION = 0;

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
		std::cout << "clCreateProgramWithSource error" << std::endl;
		return -1;
	}

	cl_int buildErr;
	buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	if (VERSION == 0) {
		cout << "Loading kernel: readWrite" << std::endl;
		kernel = clCreateKernel(program, "readWrite", &status);
	} else if (VERSION == 16) {
		cout << "Loading kernel: readWriteWithOffset16" << std::endl;
		kernel = clCreateKernel(program, "readWriteWithOffset16", &status);
	} else if (VERSION == 20) {
		cout << "Loading kernel: readWriteWithOffset20" << std::endl;
		kernel = clCreateKernel(program, "readWriteWithOffset20", &status);
	} else if (VERSION == 24) {
		cout << "Loading kernel: readWriteWithOffset24" << std::endl;
		kernel = clCreateKernel(program, "readWriteWithOffset24", &status);
	} else if (VERSION == 128) {
		cout << "Loading kernel: readWriteWithOffset128" << std::endl;
		kernel = clCreateKernel(program, "readWriteWithOffset128", &status);
	} else {
		cout << "Version not recognized\n";
		exit(-1);
	}
	if (status != CL_SUCCESS) {
		std::cout << "clCreateKernel error" << std::endl;
		return -1;
	}
	return 0;
}

ulong getOffsetVersion() {
	long offsetVersion = 0L;
	if (VERSION == 16) {
		offsetVersion = 16L;
	} else if (VERSION == 20) {
		offsetVersion = 20L;
	} else if (VERSION == 24) {
		offsetVersion = 24L;
	} else if (VERSION == 128) {
		offsetVersion = 128L;
	} else {
		cout << "Version not supported" << endl;
		exit(-1);
	}
	return offsetVersion;
}

int hostDataInitialization(int elements) {

	datasize = (sizeof(char) * 4 * elements) + VERSION;

	A = (char*) malloc(datasize);
	B = (char*) malloc(datasize);

	ulong offsetVersion = getOffsetVersion();
	for (int i = 0; i < elements; i++) {
	 	long offset = (i << 2) + offsetVersion;
	 	*(A + offset) = 100;
		// cout << float(*(A + offset)) << " ";
	 	*(B + offset) = 0;
	}
	cout << endl;
}

int allocateBuffersOnGPU() {
	cl_int status;
 	d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);
	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, NULL);
}

int writeBuffer() {
	clEnqueueWriteBuffer(commandQueue, d_A, CL_TRUE, 0, datasize, A, 0, NULL, &writeEvent1);
}

int runKernel() {
	cl_int status;
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	if (status != CL_SUCCESS) {
		std::cout << "Error in clSetKernelArg#0\n";
	}
	status  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	if (status != CL_SUCCESS) {
		std::cout << "Error in clSetKernelArg#0\n";
	}
	// launch kernel 
	size_t globalWorkSize[2];
	globalWorkSize[0] = elements;
	cl_event waitEventsKernel[] = {writeEvent1};
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 1, waitEventsKernel, &kernelEvent);
	if (status != CL_SUCCESS) {
		std::cout << "Error in clEnqueueNDRangeKernel\n";
	}

	kernelTime = getTime(kernelEvent);

	// Read result back from device to host	
	cl_event waitEventsRead[] = {kernelEvent};
	status = clEnqueueReadBuffer(commandQueue, d_B, CL_TRUE, 0,  datasize, B, 1, waitEventsKernel, &readEvent1);
	if (status != CL_SUCCESS) {
		std::cout << "Error in clEnqueueReadBuffer\n";
	}
}

void freeMemory() {
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
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
	cout << "\t -v <kernel offset>     Select kernel offset: {  0, 16, 20, 24, 128 }\n"; 
}

void processCommandLineOptions(int argc, char **argv) {
	int option;
	bool doHelp = false;
	std::set<int> allowedValues;
	allowedValues.insert(0);
	allowedValues.insert(16);
	allowedValues.insert(20);
	allowedValues.insert(24);
	allowedValues.insert(128);
	set<int>::iterator it;
	while ((option = getopt(argc, argv, ":p:s:v:h")) != -1) {
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
			case 'v':
				VERSION = atoi(optarg);
				it = allowedValues.find(VERSION);
				if (it == allowedValues.end()) {
					cout << "Value not allowed: " << VERSION << std::endl;
					exit(0);
				}
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

	cout << "Read/Write Benchmark " << endl;
	cout << "Size = " << elements << endl;

	vector<long> kernelTimers;
	vector<long> writeTimers;
	vector<long> readTimers;
	vector<double> totalTime;

	int status = openclInitialization();
	if (status < 0) {
		return -1;
	}
	hostDataInitialization(elements);
	allocateBuffersOnGPU();

	for (ulong i = 0; i < 100; i++) {

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

		// Print info ocl timers
		cout << "Iteration: " << i << endl;
		cout << "Write    : " <<  writeTime  << endl;
		cout << "X        : " <<  kernelTime  << endl;
		cout << "Reading  : " <<  readTime  << endl;
		cout << "C++ total: " << total << endl;
		cout << "\n";

		if (CHECK_RESULT) {
			bool valid = true;
			for (int i = 0; i < elements; i++) {
				long offsetVersion = getOffsetVersion();
				long offset = (i << 2) + offsetVersion;
	 			float valA = *(A + offset);
				float valB = *(B + offset);
				//cout << valB << " ";
				if (valA != valB) {
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

