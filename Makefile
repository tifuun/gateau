#### Stage find: find CUDA ####

# Candidate CUDA root directories
CUDA_CANDIDATES := \
    /usr/local/cuda \
    /usr/local/cuda-12.0 \
    /usr/local/cuda-11.8 \
    /opt/cuda \
    /usr \
    /usr/lib \
    /lib

# Try to detect where libcudart or another CUDA library is
CUDA_LIB_SEARCH := lib/libcudart.so lib64/libcudart.so

# Find a CUDA root containing a CUDA runtime library
CUDA_HOME := $(firstword $(foreach dir,$(CUDA_CANDIDATES), \
                $(foreach lib,$(CUDA_LIB_SEARCH), \
                    $(if $(wildcard $(dir)/$(lib)),$(dir),) )))

# If nothing was found, fallback or error
ifeq ($(CUDA_HOME),)
  $(warning CUDA not found in common locations)
  CUDA_HOME := /usr/local/cuda  # fallback or remove this and force error
endif

# Derive include/lib paths
CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64
# Optionally search for lib as well:
ifneq ($(wildcard $(CUDA_HOME)/lib),)
  CUDA_LIB := $(CUDA_HOME)/lib
endif

GATEAU_VERSION := 0.1.6
PYTHON         := python3

NVCC        := $(CUDA_HOME)/bin/nvcc

INCLUDES    := -Isrc/gateau/include \
               -Isrc/gateau/cuda

NVCCFLAGS   := -Xcompiler -fPIC -shared

LDFLAGS     := -L$(CUDA_LIB)
LDLIBS      := -lcfitsio -lgsl -lgslcblas -lcufft_static -lcudart_static -lculibos

CU_SOURCES  := src/gateau/cuda/kernels.cu

TARGET_LIB    := src/gateau/libgateau.so
TARGET_WHEEL_WRONG  := dist/gateau-$(GATEAU_VERSION)-py3-none-any.whl
TARGET_WHEEL  := dist/gateau-$(GATEAU_VERSION)-cp39.cp310.cp311.cp312.cp313-none-manylinux_2_31_x86_64.whl

# Default target
wheel: $(TARGET_WHEEL)

lib: $(TARGET_LIB)

$(TARGET_WHEEL): $(TARGET_LIB)
	#./scripts/wheelrename.sh
	$(PYTHON) -m build
	mv $(TARGET_WHEEL_WRONG) $(TARGET_WHEEL)

$(TARGET_LIB): $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LDLIBS)

rename:
	mv $(TARGET_WHEEL_WRONG) $(TARGET_WHEEL)

clean:
	rm -rf $(TARGET_LIB) *.o dist

