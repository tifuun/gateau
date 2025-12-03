CUDA_HOME      := /usr/local/cuda
GATEAU_VERSION := 0.1.6
PYTHON         := python3

CUDA_LIB64  := $(CUDA_HOME)/lib64
NVCC        := $(CUDA_HOME)/bin/nvcc

INCLUDES    := -Isrc/gateau/include \
               -Isrc/gateau/cuda

NVCCFLAGS   := -Xcompiler -fPIC -shared

LDFLAGS     := -L$(CUDA_LIB64)
LDLIBS      := -lgsl -lgslcblas -lcufft_static -lcudart_static -lculibos

CU_SOURCES  := src/gateau/cuda/kernels.cu

TARGET_LIB    := src/gateau/libgateau.so
TARGET_WHEEL_WRONG  := dist/gateau-$(GATEAU_VERSION)-py3-none-any.whl
TARGET_WHEEL  := dist/gateau-$(GATEAU_VERSION)-cp39.cp310.cp311.cp312.cp313-none-manylinux_2_31_x86_64.whl

# Default target
all: $(TARGET_WHEEL)

lib: $(TARGET_LIB)

$(TARGET_WHEEL): $(TARGET_WHEEL_WRONG)
	#./scripts/wheelrename.sh
	cp --reflink=auto $(TARGET_WHEEL_WRONG) $(TARGET_WHEEL)

$(TARGET_WHEEL_WRONG): $(TARGET_LIB)
	$(PYTHON) -m build

$(TARGET_LIB): $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET_LIB) *.o

