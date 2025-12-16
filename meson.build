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
cuda = dependency(
  'cuda',
  required: true,
  version : '>=11, <=13',
  modules : ['cufft']
  )

# WILL NOT WORK ON 13.0 IT WILL COMPILE BUT
# WILL CRASH AT RUNTIME WITH
# OSError: /play/gateau/build/cp313/libgateau.so: undefined symbol: cufftExecC2R
# BECAUSE OF THIS LINE IN kernels.cu:
# CUFFT_CALL( cufftExecC2R(plan, d_onef_out, d_sigout) );
# have not tested on which exact sub-versions of 11 and 12 work
# tho TODO

# Meson sets up nvcc automatically and uses its own rules.

# Extra CUDA compilation flags
cuda_args = [
  '-Xcompiler',
  '-shared',
  '-lcufft'
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
  pic: true, # no clue, something to do with static linking?
  cpp_args: cuda_args,

  # Install the .so into the gateau python module
  # directory alongside the .py files
  install: true,
  install_dir: py.get_install_dir() / 'gateau',
)

subdir('src/gateau')

