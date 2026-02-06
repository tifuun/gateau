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


cpp_args = [
  '-Xcompiler',
  '-shared',
  '-fPIC',
]

gateau_inc_dirs = [
  'src/gateau/include',
  'src/gateau/cuda',
]

cu_sources = files(
  'src/gateau/cuda/kernels.cu',
)

if host_machine.system() == 'windows'
  # lib dirs must be absolute, inc dirs must be relative?

  gsl_lib_dirs = [
    'Z:\\win\\tmp\\extracted\\gsl\\build\\native\\',
    ]

  gsl_inc_dirs = [
    'win\\tmp\\extracted\\gsl\\build\\native\\',
    ]

  hdf5_lib_dirs = [
    'Z:\\win\\tmp\\extracted\\hdf5\\hdf5\\HDF5-2.0.0-win64\\HDF5-2.0.0-win64\\lib',
    ]

  hdf5_inc_dirs = [
    'win\\tmp\\extracted\\hdf5\\hdf5\\HDF5-2.0.0-win64\\HDF5-2.0.0-win64\\include',
    ]


  cufft = dependency(
    'cuda',
    required: true,
    #version : '>=13',
    modules: ['cufft'],
    static: false,
    )

  gsl = cpp.find_library(
    'gsl',
    required: true,
    static: true,
    dirs: gsl_lib_dirs,
    )

  gslcblas = cpp.find_library(
    'gslcblas',
    required: true,
    static: true,
    dirs: gsl_lib_dirs
    )

  # On windows this MUST be cpp.find_library
  hdf5 = cpp.find_library(
    'hdf5',
    required: true,
    static: true,
    dirs: hdf5_lib_dirs,
    )

  cuda_args = [
    '-Xcompiler',
    '/std:c++14',
    '-DGATEAU_WIN_BUILD',
    ]

  add_project_arguments(cuda_args, language: 'cuda')

  zlib = cpp.find_library(
    'zlib',
    #version: '>=1.2.8',
    static: true,
    required: true,
    dirs: [
      'Z:\\win\\tmp\\extracted\\zlib\\zlib-win-x64'
      ]
    )

  aec = cpp.find_library(
    'szip',
    static: true,
    required: true,
    dirs: [
      'Z:\\win\\tmp\\extracted\\aec\\libaec-1.0.4-h33f27b4_1\\Library\\lib',
      ]
    )

  deps = [cufft, gsl, gslcblas, hdf5, zlib, aec]
  incls = [
    gateau_inc_dirs,
    gsl_inc_dirs,
    hdf5_inc_dirs,
    ]

else

  gsl = cpp.find_library(
    'gsl',
    required: true,
    static: false,
    )

  gslcblas = cpp.find_library(
    'gslcblas',
    required: true,
    static: false,
    )

  # On linux this MUST be dependency,
  # NOT cpp.find_library
  hdf5 = dependency(
    'hdf5',
    required: true,
    static: false,
    )

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

  deps = [cuda_libs, gsl, hdf5]
  incls = [
    gateau_inc_dirs,
    ]

endif

if host_machine.system() == 'windows'
  name_suffix = 'dll.a'
  # Something later on down the chain
  # expects it to be .dll.a not .dll
else
  name_suffix = 'so'
endif


libgateau = library(
  'gateau',
  cu_sources,
  include_directories: include_directories(incls),
  dependencies: deps,
  name_suffix: name_suffix,
  pic: true, # no clue, something to do with static linking?
  cpp_args: cpp_args,

  # Install the .so into the gateau python module
  # directory alongside the .py files
  install: true,
  install_dir: py.get_install_dir() / 'gateau',
)


subdir('src/gateau')

