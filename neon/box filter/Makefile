SUBDIR := .

OUTPUT := ./obj
TARGET := test
INCS := $(foreach dir,$(SUBDIR),-I$(dir))
INCS += `pkg-config --cflags opencv4` # 通过pkg-config指令,将opencv4的头文件路径列出

SRCS := $(foreach dir,$(SUBDIR),$(wildcard $(dir)/*.cpp))

OBJS := $(patsubst %.cpp,$(OUTPUT)/%.o,$(SRCS))
DEPS := $(patsubst %.o,%.d,$(OBJS))

LIBS := `pkg-config --libs opencv4` # 通过pkg-config指令,将opencv4的库文件路径列出

$(TARGET) : $(OBJS)
		@echo linking...
		@g++ -g $^ -o $@ $(LIBS)
		@echo complete!

$(OUTPUT)/%.o : %.cpp
		@echo compling...
		@mkdir -p $(dir $@)
		@g++ -g -MMD -MP -c $(INCS) $< -o $@
		@echo complete!

.PHONY : clean

clean :
		@echo cleaning...
		@rm -rf $(OUTPUT) $(TARGET)
		@echo complete!

-include $(DEPS) 