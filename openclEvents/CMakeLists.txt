cmake_minimum_required(VERSION 3.10)
project(openclEvents)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lOpenCL -fopenmp ")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(openclEvents main.cpp)