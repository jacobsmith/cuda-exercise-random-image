# Compiler and flags
NVCC = nvcc
CXX = g++
CUDA_PATH ?= /usr/local/cuda-11.3
NVCC_FLAGS = -O3 -arch=sm_86 -std=c++14
CXX_FLAGS = -O3 -std=c++14 -fPIC -I$(CUDA_PATH)/include
# Adjust -arch based on your GPU (sm_75 for Turing, sm_86 for Ampere, etc.)

# Directories
SRC_DIR = .
BUILD_DIR = build

# Files
TARGET = recolor
CU_SOURCES = recolor.cu
CPP_SOURCES = image_io.cpp
HEADERS = recolor.h

# Libraries (if using stb_image or OpenCV)
# For stb_image (header-only, no linking needed)
INCLUDES = -I.

# For OpenCV (uncomment if using)
# OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`
# NVCC_FLAGS += $(OPENCV_FLAGS)

# Rules
all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CU_SOURCES) $(CPP_SOURCES) $(HEADERS)
	$(CXX) $(CXX_FLAGS) -c $(CPP_SOURCES) -o $(BUILD_DIR)/image_io.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CU_SOURCES) $(BUILD_DIR)/image_io.o -o $(BUILD_DIR)/$(TARGET)

run: $(TARGET)
	./$(BUILD_DIR)/$(TARGET) input.jpg output.jpg

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean run