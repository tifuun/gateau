project(
  'gateau',
  'cpp', 'cuda', 'c',
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
  required: false,
  #version : '>=13',
  version : '>=11, <=13',
  modules: ['cufft'],
  )

if cuda.found()
else
  warning('Meson could not find cuda dependency, trying random stuff until it works!!!!!!')

  cuda = cc.find_library(
    'cufft',
    dirs: [
      # Should be here:
      '/usr/local/cuda/lib64',

      # Arend's laptop:
      '/usr/lib/x86_64-linux-gnu/',

      # Unusual paths, but plausible
      '/lib/x86_64-linux-gnu/',
      '/usr/lib/cuda/lib64/',
      '/usr/lib/cuda/lib64/',
      '/usr/lib/nvidia/',
      '/usr/lib64/cuda/',
      '/usr/local/cuda/lib/',
      '/usr/local/lib/',

      # Probably not here but worth checking
      '/lib64/',
      '/opt/NVIDIA/cuda/lib64/',
      '/opt/cuda/lib64/',
      '/opt/cudnn/lib64/',

      # what
      '/var/cache/cuda/lib64/',
      '/var/spool/cuda/lib64/',
      '~/.local/lib/',
      '~/.local/lib64/',
      '~/cuda/lib64/',
      ],
    required: true
    )
endif


# WILL NOT WORK ON 13.0 IT WILL COMPILE BUT
# WILL CRASH AT RUNTIME WITH
# OSError: /play/gateau/build/cp313/libgateau.so: undefined symbol: cufftExecC2R
# BECAUSE OF THIS LINE IN kernels.cu:
# CUFFT_CALL( cufftExecC2R(plan, d_onef_out, d_sigout) );
# have not tested on which exact sub-versions of 11 and 12 work
# tho TODO

# Meson sets up nvcc automatically and uses its own rules.

cpp_args = [
  '-Xcompiler',
  '-shared',
]

# ---- Include directories ----
inc_dirs = include_directories(
  'src/gateau/include',
  'src/gateau/cuda',
)

# ---- External libraries ----
gsl        = dependency('gsl', required: true)
hdf5        = dependency('hdf5', required: true)
gslcblas   = cc.find_library('gslcblas', required: true)

libs = [cuda, gsl, gslcblas, hdf5]

# ---- Sources ----
cu_sources = files(
  'src/gateau/cuda/kernels.cu',
)

# ---- Build shared library ----
libgateau = library(
  'gateau',
  cu_sources,
  include_directories: inc_dirs,
  dependencies: libs,
  name_suffix: 'so',
  pic: true, # no clue, something to do with static linking?
  cpp_args: cpp_args,

  # Install the .so into the gateau python module
  # directory alongside the .py files
  install: true,
  install_dir: py.get_install_dir() / 'gateau',
)

subdir('src/gateau')

