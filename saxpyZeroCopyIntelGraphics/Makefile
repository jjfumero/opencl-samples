all: iocl  

iocl:
	g++ -c ../common/readSource.cpp  -O3
	g++ -std=c++0x saxpy.cpp readSource.o -o saxpy -lOpenCL -fopenmp -O3

nvidia:
	g++ -c ../common/readSource.cpp  -O3
	nvcc  -std=c++11 saxpy.cpp -o nvidia -lOpenCL readSource.o -O3
	
clean:
	rm *.o saxpy
