SRC := ./wmma_v1.cu
TAR := ./wmma_v1

CC = /home/xbj/miniconda3/envs/cudax/bin/nvcc
LD_FLAGS := -lcublas -lm
# ARCH := -arch=sm_70
GENCODE := -gencode=arch=compute_70,code=\"sm_70,compute_70,sm_75,sm_86\"
$(TAR) : $(SRC)
	@$(CC)  $< -o $@ $(LD_FLAGS) $(GENCODE)

.PHONY : clean

clean :
	@rm -rf $(TAR)