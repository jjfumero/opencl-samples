#include <iostream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <algorithm>
using namespace std;

#define CL_TARGET_OPENCL_VERSION 200

const bool CHECK_RESULT = true;

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

char *readsource(const char *sourceFilename) {

    FILE *fp;
    int err;
    int size;
    char *source;

    fp = fopen(sourceFilename, "rb");

    if(fp == NULL) {
        printf("Could not open kernel file: %s\n", sourceFilename);
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_END);

    if(err != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);

    }
    size = ftell(fp);

    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_SET);
    if(err != 0) {
        printf("Error seeking to start of file\n");
        exit(-1);

    }

    source = (char*)malloc(size+1);

    if(source == NULL) {
        printf("Error allocating %d bytes for the program source\n", size+1);
        exit(-1);
    }

    err = fread(source, 1, size, fp);
    if(err != size) {
        printf("only read %d bytes\n", err);
        exit(0);
    }

    source[size] = '\0';
    return source;
}



class SaxpyApp {

    const int PLATFORM_ID = 0;
    int elements = 1024 * 1024 * 64;
    double alpha;

    size_t datasize;
    double *A;
    double *B;
    double *C;

    string platformName;
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
    cl_mem ddA;
    cl_mem ddB;
    cl_mem ddC;

    cl_event kernelEvent = NULL;
    cl_event writeEvent1  = NULL;
    cl_event writeEvent2  = NULL;
    cl_event readEvent1  = NULL;

    long kernelTime;
    long writeTime;
    long readTime;

public:
    long getElapsedTime(cl_event event);
    int openclInitialization();
    int hostDataInitialization(int elements);
    int allocateBuffersOnGPU();
    int writeBuffer();
    int runKernel();
    void freeMemory();
    double median(vector<long> data);
    double median(vector<double> data);
    int run(int argc, char **argv);

    int readBuffer();
};

long SaxpyApp::getElapsedTime(cl_event event) {
    clWaitForEvents(1, &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    return (time_end - time_start);
}

int SaxpyApp::openclInitialization() {

    cl_int status;
    cl_uint numPlatforms = 0;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    if (numPlatforms == 0) {
        cout << "No platform detected" << endl;
        return -1;
    }

    platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
    if (platforms == NULL) {
        cout << "malloc platform_id failed" << endl;
        return -1;
    }

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        cout << "clGetPlatformIDs failed" << endl;
        return -1;
    }

    cout << numPlatforms << " has been detected" << endl;
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
        std::cout << "Error with clGetDeviceIDs\n";
    }

    if (status != CL_SUCCESS) {
        cout << "[WARNING] Using CPU, no GPU available" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    } else {
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }

    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if (context == NULL) {
        cout << "Context is not NULL" << endl;
    }

    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    //commandQueue = clCreateCommandQueue(context, devices[0], NULL, &status);
    if (status != CL_SUCCESS || commandQueue == NULL) {
        cout << "Error in create command" << endl;
        return -1;
    }

    const char *sourceFile = "kernel.cl";
    source = readsource(sourceFile);
    program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &status);
    if (status != CL_SUCCESS) {
        std::cout << "Error with clCreateProgramWithSource\n";
        std::terminate();
    }

    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        std::cout << "Error with clBuildProgram\n";
        std::terminate();
    }
    kernel = clCreateKernel(program, "saxpy", &status);
}

int SaxpyApp::hostDataInitialization(int elements) {
    datasize = sizeof(double) * elements;
    alpha = 12.0f;

    ddA = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, datasize, NULL, NULL);
    ddB = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, datasize, NULL, NULL);
    ddC = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, datasize, NULL, NULL);

    A = (double *) clEnqueueMapBuffer(commandQueue, ddA, CL_TRUE, CL_MAP_WRITE, 0, datasize, 0, NULL, NULL, NULL);
    B = (double *) clEnqueueMapBuffer(commandQueue, ddB, CL_TRUE, CL_MAP_WRITE, 0, datasize, 0, NULL, NULL, NULL);
    C = (double *) clEnqueueMapBuffer(commandQueue, ddC, CL_TRUE, CL_MAP_READ, 0, datasize, 0, NULL, NULL, NULL);

    #pragma omp parallel for
    for (int i = 0; i < elements; i++) {
        A[i] = 2;
        B[i] = 4;
    }
}

int SaxpyApp::allocateBuffersOnGPU() {
    cl_int status;
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, elements * sizeof(double), NULL, &status);
    if (status != CL_SUCCESS) {
        cout << "Error with clCreateBuffer (A)\n";
    }
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, elements * sizeof(double), NULL, &status);
    if (status != CL_SUCCESS) {
        cout << "Error with clCreateBuffer (B)\n";
    }
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, elements * sizeof(double), NULL, &status);
    if (status != CL_SUCCESS) {
        cout << "Error with clCreateBuffer (C)\n";
    }
}

int SaxpyApp::writeBuffer() {
    // Create write events
    cl_int status = clEnqueueWriteBuffer(commandQueue, d_A, CL_FALSE, 0, elements * sizeof(double), A, 0, NULL, &writeEvent1);
    status |= clEnqueueWriteBuffer(commandQueue, d_B, CL_FALSE, 0, elements * sizeof(double), B, 0, NULL, &writeEvent2);
    return status;
}

int SaxpyApp::runKernel() {
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_double), &alpha);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_int), &elements);

    // launch kernel
    size_t globalWorkSize[1];
    globalWorkSize[0] = elements;
    cl_event waitEventsKernel[] = {writeEvent1, writeEvent2};
    status |= clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 2, waitEventsKernel, &kernelEvent);

    //kernelTime = getElapsedTime(kernelEvent);
    return status;
}

int SaxpyApp::readBuffer() {
    // Read result back from device to host
    cl_event waitEventsRead[] = {kernelEvent};
    cl_int status = clEnqueueReadBuffer(commandQueue, d_C, CL_TRUE, 0, sizeof(double) * elements, C, 1, waitEventsRead, &readEvent1);
    return status;
}

void SaxpyApp::freeMemory() {
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

double SaxpyApp::median(vector<long> data) {
    if (data.empty()) {
        return 0;
    } else {
        sort(data.begin(), data.end());
        if (data.size() % 2 == 0) {
            return (data[data.size() / 2 - 1] + data[data.size() / 2]) / 2;
        } else {
            return double(data[data.size() / 2]);
        }
    }
}

double SaxpyApp::median(vector<double> data) {
    if (data.empty()) {
        return 0;
    } else {
        sort(data.begin(), data.end());
        if (data.size() % 2 == 0) {
            return (data[data.size() / 2 - 1] + data[data.size() / 2]) / 2;
        } else {
            return double(data[data.size() / 2]);
        }
    }
}

int SaxpyApp::run(int argc, char **argv) {

    if (argc > 1) {
        elements = atoi(argv[1]);
    }

    cout << "OpenCL Saxpy " << endl;
    cout << "Size = " << elements << endl;

    vector<long> kernelTimers;
    vector<long> writeTimers;
    vector<long> readTimers;
    vector<double> totalTime;

    openclInitialization();
    hostDataInitialization(elements);
    allocateBuffersOnGPU();

    for (int i = 0; i < 5; i++) {

        kernelTime = 0;
        writeTime = 0;
        readTime = 0;

        auto start_time = chrono::high_resolution_clock::now();
        cout << "[BEFORE] " << writeEvent1 << endl;

        writeBuffer();
        runKernel();
        readBuffer();
        auto end_time = chrono::high_resolution_clock::now();

        writeTime = getElapsedTime(writeEvent1);
        writeTime += getElapsedTime(writeEvent2);
        kernelTime = getElapsedTime(kernelEvent);
        readTime = getElapsedTime(readEvent1);

        clReleaseEvent(writeEvent1);
        clReleaseEvent(writeEvent2);
        clReleaseEvent(kernelEvent);
        clReleaseEvent(readEvent1);

        cout << "[AFTER] " << writeEvent1 << endl;

        kernelTimers.push_back(kernelTime);
        writeTimers.push_back(writeTime);
        readTimers.push_back(readTime);

        double total = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
        totalTime.push_back(total);

        if (CHECK_RESULT) {
            bool valid = true;
            for (int i = 0; i < elements; i++) {
                if (C[i] != (alpha * A[i]) + B[i]) {
                    cout << C[i] << "  != " << (alpha * A[i]) + B[i] << " ::IDX: " << i << endl;
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

        // Print info ocl timers
        cout << "Iteration: " << i << endl;
        cout << "Write    : " << writeTime << endl;
        cout << "X        : " << kernelTime << endl;
        cout << "Reading  : " << readTime << endl;
        cout << "C++ total: " << total << endl;
        cout << "\n";
    }

    freeMemory();

    // Compute median values
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

int main(int argc, char **argv) {
    SaxpyApp saxpyApp;
    saxpyApp.run(argc, argv);
    return 0;
}



