project(
  'gateau',
  'cpp', 'c',
  version: '0.1.6',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17'
  ]
)

# ------------------------------------------------------------
# Step One: Try to find NVCC cuda compiler
# ------------------------------------------------------------
 
# First, try to find it using meson's builtin logic

meson_could_find_cuda = add_languages('cuda', required: false, native: true)

# `native` flag means we want to use it to compile stuff
# for THIS machine. We don't support cross-compilation,
# but still set the flag to silence meson warning.

if meson_could_find_cuda
  message('Meson could find nvcc compiler normally, yay!')
  add_languages('cuda', native: true, required: true)
else
  error('Meson could not find nvcc. Make sure it is in your path.')
  #warning('Meson could not find NVCC compiler!!')
  #warning('Is `nvcc` in your $PATH?')
  #warning('Trying random paths until something works...')

  ## If meson's nvcc detection logic fails, try some common paths

  #cuda_search_paths = [
  #  '/usr/local/cuda/bin',
  #  '/opt/cuda/bin',
  #  '/usr/local/cuda-12.0/bin',
  #  '/usr/local/cuda-11.8/bin',
  #]

  #nvcc = find_program(
  #  'nvcc',
  #  dirs: cuda_search_paths,
  #  required: false
  #)

  #if not nvcc.found()
  #  error('Could not find NVCC, giving up.')
  #endif

  #message('Found nvcc at: ' + nvcc.path())

  #run_command(
  #  [
  #    'ln',
  #    '-sf',
  #    nvcc.path(),
  #    './nvcc',
  #    ]
  #  )
  #run_command(
  #  ['sh', '-c', 'env > /tmp/foo'],
  #  )
  #nvcc_path = nvcc.full_path()
  #run_command(
  #  ['sh', '-c', 'ln -sf "' + nvcc_path + '" "$VIRTUAL_ENV/bin/nvcc"']
  #  )

  # ----------------------------------------------------------
  # 3. Override Meson’s nvcc lookup and retry CUDA
  # ----------------------------------------------------------
  #meson.override_find_program('cuda', nvcc)
  #
  #env = environment()
  #env.set('PATH', '/usr/local/cuda/bin')
  #meson.add_env(env)

  add_languages('cuda', native: true, required: true)
endif

# ------------------------------------------------------------
# Step Two: Try to find Cuda libraries
# ------------------------------------------------------------

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
cuda = meson.get_compiler('cuda')

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
gsl = dependency('gsl', required: true)
hdf5 = dependency('hdf5', required: true)
gslcblas = cpp.find_library('gslcblas', required: true)

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

