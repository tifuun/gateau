project(
  'gateau',
  'cpp', 'cuda', 'c',
  version: '0.1.6',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17',
    'c_std=c17',
  ]
)

cpp = meson.get_compiler('cpp')
py = import('python').find_installation()

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

# ---- figure out static/dynamic gsl+hdf5 ----

# are we building a wheel or are we `pip install`ing?
is_building_wheel = run_command(
  'sh', '-c', 'test -n "$PEP517_BUILD_BACKEND" && echo yes || echo no',
  check : false
).stdout().strip() == 'no'

if get_option('gsl_static').enabled()
  message('Option set to use STATIC gsl')
  gsl_static = true
elif get_option('gsl_static').disabled()
  message('Option set to use DYNAMIC gsl')
  gsl_static = false
else
  message('Option set to autochoose static/dynamic gsl...')

  if is_building_wheel
    message('we are building wheel, therefore choosing STATIC gsl')
    gsl_static = true
  else
    message('we are building to install, therefore choosing DYNAMIC gsl')
    gsl_static = false
  endif
endif


if get_option('hdf5_static').enabled()
  message('Option set to use STATIC hdf5')
  hdf5_static = true
elif get_option('hdf5_static').disabled()
  message('Option set to use DYNAMIC hdf5')
  hdf5_static = false
else
  message('Option set to autochoose static/dynamic hdf5...')

  if is_building_wheel
    message('we are building wheel, therefore choosing STATIC hdf5')
    hdf5_static = true
  else
    message('we are building to install, therefore choosing DYNAMIC hdf5')
    hdf5_static = false
  endif
endif

# ---- External libraries ----
gsl = dependency(
  'gsl',
  required: true,
  static: gsl_static
  )
gslcblas = cpp.find_library(
  'gslcblas',
  required: true,
  static: gsl_static
  )
hdf5 = dependency(
  'hdf5',
  required: true,
  static: hdf5_static
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

