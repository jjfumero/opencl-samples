all: iocl  

iocl:
	g++ -c ../common/readSource.cpp  -O3
	g++ -std=c++0x test01.cpp readSource.o -o test01 -lOpenCL -fopenmp -O3

gpu:
	g++ -c ../common/readSource.cpp  -O3
	nvcc  -std=c++11 test01.cpp -o test01 -lOpenCL readSource.o -O3
	
clean:
	rm *.o test01
