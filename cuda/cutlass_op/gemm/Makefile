SRC := ./sgemm_basic.cu
TARGET := ./sgemm_basic

CC := /usr/local/cuda/bin/nvcc

CXXFLAGS := -std=c++17

GENCODE := -gencode arch=compute_86,code=sm_86

INC := -I/root/cutlass/cutlass/include

$(TARGET) : $(SRC)
	@$(CC) $(INC) $(CXXFLAGS) $< -o $@ $(GENCODE) 

.PHONY : clean

clean : 
	@rm -rf $(TARGET)