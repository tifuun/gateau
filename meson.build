project(
  'gateau',
  'cuda', 'cpp', 'c',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17',
    'c_std=c17',
  ]
)

py = import('python').find_installation()

cpp = meson.get_compiler('cpp', native: true)

gsl = dependency('gsl', required: true)
gslcblas = cpp.find_library('gslcblas', required: true)
hdf5 = dependency('hdf5', required: true)

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

      # sometimes /usr/local/cuda symlink to
      # specific version doesnt get made,
      # for example on ubuntu cuda11 packages,
      # so check there manually
      '/usr/local/cuda-12/lib64',
      '/usr/local/cuda-12.9/lib64',
      '/usr/local/cuda-12.8/lib64',
      '/usr/local/cuda-12.7/lib64',
      '/usr/local/cuda-12.6/lib64',
      '/usr/local/cuda-12.5/lib64',
      '/usr/local/cuda-12.4/lib64',
      '/usr/local/cuda-12.3/lib64',
      '/usr/local/cuda-12.2/lib64',
      '/usr/local/cuda-12.1/lib64',
      '/usr/local/cuda-11/lib64',
      '/usr/local/cuda-11.9/lib64',
      '/usr/local/cuda-11.8/lib64',
      '/usr/local/cuda-11.7/lib64',
      '/usr/local/cuda-11.6/lib64',
      '/usr/local/cuda-11.5/lib64',
      '/usr/local/cuda-11.4/lib64',
      '/usr/local/cuda-11.3/lib64',
      '/usr/local/cuda-11.2/lib64',
      '/usr/local/cuda-11.1/lib64',

      # Debian official packages
      # (NOT nvidia's packages for debian)
      # put it here:
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
    error('''
      Could not find cufft library.
      Please check your cuda installation.
      If everything looks okay, try installing gateau
      from source (i.e. by cloning repo)
      and making sure that that location of your `libcufft.so`
      is included in the search paths specified in `meson.build`.
      Then, please open an issue.
      ''')
  endif
endif

cpp_args = [
  '-Xcompiler',
  '-shared',
  '-fPIC',
]

inc_dirs = include_directories(
  'src/gateau/include',
  'src/gateau/cuda',
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

subdir('src/gateau')

