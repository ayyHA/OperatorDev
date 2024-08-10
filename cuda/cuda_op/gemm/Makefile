SRC := ./sgemmV2.cu
TAR := ./sgemmV2.exe

CC = /home/xbj/miniconda3/envs/cudax/bin/nvcc
LD_FLAGS := -lcublas 


$(TAR) : $(SRC)
	@$(CC)  $< -o $@ $(LD_FLAGS)

.PHONY : clean

clean :
	@rm -rf *.exe