SRC := ./mma_naive.cu
TAR := ./mma_naive

CC = /home/xbj/miniconda3/envs/cudax/bin/nvcc
LD_FLAGS := -lcublas -lm
CXXFLAGS := -std=c++11
# ARCH := -arch=sm_70
GENCODE := -gencode arch=compute_86,code=\"sm_86\"
$(TAR) : $(SRC)
	@$(CC) $(CXXFLAGS) $< -o $@ $(LD_FLAGS) $(GENCODE)

.PHONY : clean

clean :
	@rm -rf $(TAR)