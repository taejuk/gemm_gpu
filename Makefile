NVCC = nvcc

NVCC_FLAGS = -O3 -std=c++17 -I.

TARGET = spmv_experiment

SRCS = main.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
