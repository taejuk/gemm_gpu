NVCC = nvcc

NVCC_FLAGS = -O3 -std=c++17 -I.


SRCS = spmv_experiment.cu memory_coalescing_test.cu update_test.cu

EXECS = $(SRCS:.cu=)


all: $(EXECS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(EXECS)

