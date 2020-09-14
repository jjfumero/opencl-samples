all: iocl  

iocl:
	g++ -c ../common/readSource.cpp  -O3
	g++ -std=c++0x atomics.cpp readSource.o -o atomics -lOpenCL -fopenmp -O3

nvidia:
	g++ -c ../common/readSource.cpp  -O3
	nvcc  -std=c++11 atomics.cpp -o nvidia -lOpenCL readSource.o -O3
	
clean:
	rm *.o atomics
