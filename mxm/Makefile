all: gcc 

gcc:
	g++ -c ../common/readSource.cpp 
	g++ -std=c++0x mxm.cpp readSource.o -o mxm -lOpenCL

debug:
	g++ -g -c ../common/readSource.cpp 
	g++ -g -std=c++0x mxm.cpp readSource.o -o mxm -lOpenCL


run:
	./mxm 1024

clean:
	rm *.o mxm
