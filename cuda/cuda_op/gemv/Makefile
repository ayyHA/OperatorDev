SRC := ./sgemv_v3.cu
TAR := ./sgemv_v3

CC = /home/xbj/miniconda3/envs/cudax/bin/nvcc
LD_FLAGS := -lcublas 
CFLAGS := -std=c++11

$(TAR) : $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LD_FLAGS)

.PHONY : clean

clean :
	rm -rf $(TAR)