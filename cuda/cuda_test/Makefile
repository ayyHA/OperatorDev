SRC := ./test_ldmatrix.cu
TARGET := ./test_ldmatrix

CC := /home/xbj/miniconda3/envs/cudax/bin/nvcc

CFLAGS := -std=c++11

GENCODE := -gencode arch=compute_75,code=\"sm_75,sm_86\"

$(TARGET) : $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(GENCODE) 

.PHONY : clean

clean : 
	@rm -rf $(TARGET)