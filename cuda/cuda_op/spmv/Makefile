SRC := ./spmv.cu
TAR := ./spmv

CC = /home/xbj/miniconda3/envs/cudax/bin/nvcc
LD_FLAGS := -lcublas -lm -lcusparse
# ARCH := -arch=sm_70
GENCODE := -gencode=arch=compute_70,code=\"sm_70,compute_70,sm_75,sm_86\"
CXXFLAGS := -std=c++11
$(TAR) : $(SRC)
	@$(CC) $(CXXFLAGS) $< -o $@ $(LD_FLAGS) $(GENCODE)

.PHONY : clean

clean :
	@rm -rf $(TAR)