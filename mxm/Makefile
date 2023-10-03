all: gcc 

gcc:
	g++ -c ../common/readSource.cpp 
	g++ -std=c++0x mxm.cpp readSource.o -o mxm -lOpenCL

clean:
	rm *.o mxm
