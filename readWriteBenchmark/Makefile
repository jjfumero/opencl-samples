all: gcc 

gcc:
	g++ -c ../common/readSource.cpp 
	g++ -std=c++0x read-write-benchmark.cpp readSource.o -o read-write-benchmark -lOpenCL

clean:
	rm *.o read-write-benchmark
