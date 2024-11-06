CC = /usr/local/cuda-12.1/bin/nvcc
CC_FLAGS = -rdc=true -O3 -lpthread -lnvidia-ml -arch=sm_80 -lcudnn -lcudadevrt -lcudart_static -lrt -ldl -lcublas -lpmem -Xcompiler -fopenmp -lpci
CUDA_SDK_PATH := $(SDK_DIR)
CUDA_PATH := /usr/local/cuda-12.1

INCLUDES := -I$(CUDA_PATH)/include

# Determine the correct version of the cutil library
# CUTIL_LIB =  # -lcutil
# ifeq ($(shell uname -m), x86_64)
#      ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
#              CUTIL_LIB = #-lcutil_x86_64

BUILD := ./build
CP_ITER ?= 10
SRC := ./src


build: checkp_func.o cuda_wrapper.o file_link.o libtest.so


checkp_func.o: checkp_func.cu
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $^ -c --compiler-options '-fPIC'  -I$(CUDA_SDK_PATH)/common/inc $(INCLUDES)  -rdc=true  -DCP_ITER=${CP_ITER} -DNVM_ALLOC_CPU -std=c++11

cuda_wrapper.o: cuda_wrapper.cpp
	g++  -O3 -c cuda_wrapper.cpp -fPIC $(INCLUDES) -o cuda_wrapper.o

file_link.o: checkp_func.o
	$(CC)  --compiler-options '-fPIC'  -arch=sm_80 -dlink -o file_link.o checkp_func.o -lcudadevrt -lcudart


libtest.so: cuda_wrapper.o file_link.o checkp_func.o
	g++ -shared -fPIC -o libtest.so checkp_func.o file_link.o cuda_wrapper.o $(INCLUDES) -L$(CUDA_PATH)/lib64 -lcudart -lpmem -lcudadevrt -lpci

tester: checkp_func.o cuda_wrapper.o file_link.o
	g++ -o tester checkp_func.o file_link.o cuda_wrapper.o $(INCLUDES) -L$(CUDA_PATH)/lib64 -lcudart -lpmem -lcudadevrt

clean:
	rm -f *.linkinfo
	rm -rf ${BUILD}/*
	rm -rf *.o
	rm -rf libtest.so
