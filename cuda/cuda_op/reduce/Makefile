SRC := ./reduce_shuffle.cu
TAR := ./reduce_shuffle.exe

CC = nvcc

$(TAR) : $(SRC)
	@$(CC) $< -o $@

.PHONY : clean

clean :
	@rm -rf *.exe