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

  name_suffix = 'dll.a'
  # Something later on down the chain
  # expects it to be .dll.a not .dll

  gsl_lib_dirs = [
    'Z:\\win\\tmp\\extracted\\gsl\\build\\native\\',
    ]
  gsl_inc_dirs = [
    'win\\tmp\\extracted\\gsl\\build\\native\\',
    ]
  hdf5_lib_dirs = [
    # Default install location with the msi installer:
    #'C:\\Program Files\\HDF_Group\\HDF5\\2.0.0\\lib'

    'Z:\\win\\tmp\\extracted\\hdf5\\hdf5\\HDF5-2.0.0-win64\\HDF5-2.0.0-win64\\lib',
    ]
  hdf5_inc_dirs = [
    # Default install location with the msi installer:
    #'C:\\Program Files\\HDF_Group\\HDF5\\2.0.0\\include'

    'win\\tmp\\extracted\\hdf5\\hdf5\\HDF5-2.0.0-win64\\HDF5-2.0.0-win64\\include',
    ]


  cuda_args = [
    #'-ccbin',
    #'cl',
    #'--use-local-env',
    '-Xcompiler',
    '/std:c++14',
    #'-shared',
    #'/libdir:noauto',
    '-DGATEAU_WIN_BUILD',
    ]
  add_project_arguments(cuda_args, language: 'cuda')

  zlib_dep = cpp.find_library(
    'zlib',
    #version: '>=1.2.8',
    static: false,
    required: true,
    dirs: [
      'Z:\\win\\tmp\\extracted\\zlib\\zlib-win-x64'
      ]
    )

  aec_dep = cpp.find_library(
    'szip',
    static: false,
    required: true,
    dirs: [
      'Z:\\win\\tmp\\extracted\\aec\\libaec-1.0.4-h33f27b4_1\\Library\\lib',
      ]
    )

  win_deps = [zlib_dep, aec_dep]

else
  name_suffix = 'so'
  # Something later on down the chain
  # expects it to be .dll.a not .dll

  gsl_lib_dirs = []
  gsl_inc_dirs = []
  hdf5_lib_dirs = []
  hdf5_inc_dirs = []
  win_deps = []
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

libs = [cuda_libs, gsl, gslcblas, hdf5] + win_deps

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
  name_suffix: name_suffix,
  pic: true, # no clue, something to do with static linking?
  cpp_args: cpp_args,

  # Install the .so into the gateau python module
  # directory alongside the .py files
  install: true,
  install_dir: py.get_install_dir() / 'gateau',
)


subdir('src/gateau')

