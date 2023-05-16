CC = gcc
NVCC = nvcc
CFLAGS = -Wall -g
LDFLAGS = -lm

SRCDIR = ./code
SOURCE = $(SRCDIR)/radiator_main.c $(SRCDIR)/gpu_functions.cu
HEADER = $(SRCDIR)/CUDA_functions.h

BUILDDIR = ./build
OBJECT = $(BUILDDIR)/radiator_main.o

TARGET = radiator_simulator

all: $(TARGET)

$(OBJECT): $(SOURCE) | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECT)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILDDIR):
	mkdir $(BUILDDIR)

# $(OBJS)
# 	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# %.o: %.c $(HEADERS)
# 	$(NVCC) -c $< -o $@

# clean:
# 	rm -rf $(BUILDDIR)/*.o $(TARGET)

# .PHONY: clean