project(
  'gateau',
  'cpp', 'cuda', 'c',
  version: '0.1.6',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17'
  ]
)

add_languages('cuda', required: true, native: true)

cuda_libs = dependency(
  'cuda',
  required: false,
  #version : '>=13',
  modules: ['cufft'],
  )

if cuda_libs.found()
  message('Meson could find cuda libraries normally, yay!')
else
  warning('Meson could not find CUDA libraries!!')
  warning('Trying random paths until something works...')


  cuda_libs = cpp.find_library(
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
    required: false
    )

  if not cuda_libs.found()
    error('Could not find CUDA libraries, giving up.')
  endif
endif

cpp = meson.get_compiler('cpp')
py = import('python').find_installation()

cpp_args = [
  '-Xcompiler',
  '-shared',
  '-fPIC',
]

# ---- Include directories ----
inc_dirs = include_directories(
  'src/gateau/include',
  'src/gateau/cuda',
)

# ---- External libraries ----
gsl = dependency(
  'gsl',
  required: true,
  static: !get_option(allow_dynamic)
  )
hdf5 = dependency(
  'hdf5',
  required: true,
  static: true,
  static: !get_option(allow_dynamic)
  )
gslcblas = cpp.find_library(
  'gslcblas',
  required: true,
  static: true,
  static: !get_option(allow_dynamic)
  )

libs = [cuda_libs, gsl, gslcblas, hdf5]

cu_sources = files(
  'src/gateau/cuda/kernels.cu',
)

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

#  libgateau = custom_target(
#    'gateau',
#    command: [
#      nvcc,
#      '-Xcompiler',
#      '-shared',
#      '-I', '../src/gateau/include',
#      '-I', '@BUILD_ROOT@/src/gateau/cuda',
#      '@INPUT@',
#      ],
#
#    depend_files: [
#      'src/gateau/include',
#      ],
#    input: cu_sources,
#    output: ['libgateau.so'],
#
#    # Install the .so into the gateau python module
#    # directory alongside the .py files
#    install: true,
#    install_dir: py.get_install_dir() / 'gateau',
#  )
#
#endif

subdir('src/gateau')

