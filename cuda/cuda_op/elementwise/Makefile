SOURCE := ./elementwise_add_float.cu
TARGET := ./elementwise_add_float

CC := /home/xbj/miniconda3/bin/nvcc
CFLAGS := -std=c++11

INC := -I/home/xbj/miniconda3/envs/cudax/include

$(TARGET) : $(SOURCE)
	@$(CC) $(CFLAGS) $(INC) $< -o $@

.PHONY : clean

clean :
	@rm -rf $(TARGET)