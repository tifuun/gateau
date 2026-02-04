project(
  'gateau',
  'cpp', 'cuda', 'c',
  version: '0.2.0',
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
gateau_inc_dirs = [
  'src/gateau/include',
  'src/gateau/cuda',
]

# ---- figure out static/dynamic gsl+hdf5 ----

gsl_static = false
hdf5_static = false

# ---- External libraries ----

if host_machine.system() == 'windows'
  gsl_lib_dirs = [
    #'Z:\\win\\dep\\gsl-msvc14-x64.2.3.0.2779\\build\\native\\static\\gsl.lib',
    #'Z:\\win\\dep\\gsl-msvc14-x64.2.3.0.2779\\build\\native\\static\\gslcblas.lib',
    'Z:\\win\\dep\\gsl-msvc14-x64.2.3.0.2779\\build\\native',
    ]
  gsl_inc_dirs = [
    'win\\dep\\gsl-msvc14-x64.2.3.0.2779\\build\\native',
    ]
  hdf5_lib_dirs = [
    'win\\dep\\hdf5\\HDF5-2.0.0-win64\\lib',
    ]
  hdf5_inc_dirs = [
    'Z:\\win\\dep\\hdf5\\HDF5-2.0.0-win64\\include',
    ]
else
  gsl_lib_dirs = []
  gsl_inc_dirs = []
  hdf5_lib_dirs = []
  hdf5_inc_dirs = []
endif

gsl = cpp.find_library(
  'gsl',
  required: true,
  static: gsl_static,
  dirs: gsl_lib_dirs,
  )
gslcblas = cpp.find_library(
  'gslcblas',
  required: true,
  static: gsl_static,
  dirs: gsl_lib_dirs
  )
hdf5 = cpp.find_library(
  'hdf5',
  required: true,
  static: hdf5_static,
  dirs: hdf5_lib_dirs,
  )

libs = [cuda_libs, gsl, gslcblas, hdf5]

cu_sources = files(
  'src/gateau/cuda/kernels.cu',
)

libgateau = library(
  'gateau',
  cu_sources,
  include_directories: include_directories(
    gateau_inc_dirs,
    gsl_inc_dirs,
    hdf5_inc_dirs,
    ),
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

