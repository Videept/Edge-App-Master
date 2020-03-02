GPU=1
CUDNN=1
OPENCV=0
OPENMP=0
DEBUG=0
# locations of cuda libraries etc, update as needed to match local config
CUDABASE_PATH=/usr/local/cuda
LIBCUDA_PATH=$(CUDABASE_PATH)/lib64 
LIBCUDA_INCLUDE_PATH=$(CUDABASE_PATH)/include
NVCC=$(CUDABASE_PATH)/bin/nvcc
export GPU CUDNN OPENCV OPENMP DEBUG LIBCUDA_PATH LIBCUDA_INCLUDE_PATH NVCC # pass these settings on to darknet

LIBJPEG_TURBO=1

MAKE=make
CC=gcc
EXEC=server

CFLAGS=-Ofast -g
LDFLAGS=-lm -ldarknet -lpthread -Ldarknet -Wl,-rpath=./darknet -Wl,-rpath=$(LIBCUDA_PATH)
ifeq ($(LIBJPEG_TURBO), 1)
CFLAGS+= -DLIBJPEG -Ilibjpeg-turbo/include
LDFLAGS+= -Llibjpeg-turbo/lib64 -ljpeg -Wl,-rpath=./libjpeg-turbo/lib64
endif

all:
	cd darknet && $(MAKE) -e && cd .. && make server 

server: $(EXEC).o Makefile darknet/include/darknet.h darknet/libdarknet.so 
	$(CC) -o $(EXEC) $(EXEC).o $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf $(EXEC) && cd darknet && $(MAKE) clean
