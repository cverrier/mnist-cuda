CC = clang
NVCC = nvcc
CFLAGS = -std=c23 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wdouble-promotion -Wformat=2 -O0 -g
NVCCFLAGS = -O2 -g

# CUDA paths (adjust for your system)
CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

all: bin/display_digits bin/test_bswap

bin/display_digits: display_digits.c mnist.c mnist.h bswap.c bswap.h display.c display.h | bin
	$(CC) $(CFLAGS) display_digits.c mnist.c bswap.c display.c -o $@

# CUDA model object file
bin/model.o: model.cu model.h | bin
	$(NVCC) $(NVCCFLAGS) -c model.cu -o $@

# Test for model (links C and CUDA)
bin/test_model: test_model.c bin/model.o model.h | bin
	$(CC) $(CFLAGS) $(CUDA_INCLUDE) test_model.c bin/model.o $(CUDA_LIBS) -o $@

# Training executable (CUDA + C sources compiled together with nvcc)
bin/train: train.cu bin/model.o mnist.c bswap.c model.h mnist.h | bin
	$(NVCC) $(NVCCFLAGS) train.cu mnist.c bswap.c bin/model.o $(CUDA_LIBS) -o $@

bin/test_bswap: test_bswap.c bswap.c bswap.h | bin
	$(CC) $(CFLAGS) test_bswap.c bswap.c -o $@

bin/test_mnist: test_mnist.c mnist.c mnist.h bswap.c bswap.h | bin
	$(CC) $(CFLAGS) test_mnist.c mnist.c bswap.c -o $@

display: bin/display_digits
	./bin/display_digits

bin:
	mkdir -p bin

test: bin/test_bswap bin/test_mnist
	./bin/test_bswap
	./bin/test_mnist

test-cuda: bin/test_model
	./bin/test_model

train: bin/train
	./bin/train

clean:
	rm -rf bin

.PHONY: all display test test-cuda train clean
