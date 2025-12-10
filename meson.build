project(
  'gateau',
  'cpp', 'cuda',
  version: '0.1.6',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17'
  ]
)

cc = meson.get_compiler('cpp')
py = import('python').find_installation()

# ---- CUDA ----
cuda = dependency('cuda', required: true)

# Meson sets up nvcc automatically and uses its own rules.

# Extra CUDA compilation flags
cuda_args = [
  '-Xcompiler', '-fPIC',
  '-shared',
]

# ---- Include directories ----
inc_dirs = include_directories(
  'src/gateau/include',
  'src/gateau/cuda',
  '/usr/include/hdf5/serial',
)

# ---- External libraries ----
gsl        = dependency('gsl', required: true)
gslcblas   = cc.find_library('gslcblas', required: true)

cuda_libs = [gsl, gslcblas]

# ---- Sources ----
cu_sources = files(
  'src/gateau/cuda/kernels.cu',
)

# ---- Build shared library ----
libgateau = library(
  'gateau',
  cu_sources,
  include_directories: inc_dirs,
  dependencies: [cuda] + cuda_libs,
  name_suffix: 'so',
  install: false,  # Set to true if you want installation
  cpp_args: cuda_args,
)

subdir('src/gateau')

