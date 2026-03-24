# Info for maintainers

Info for maintainers is available in the docs (
[html](https://tifuun.github.io/gateau/maintainers.html),
[source](./doxy/maintainers.dox)
)


---

- sudo apt-get install gcc g++ pkg-config libgsl-dev libhdf5-dev

---

ubuntu no cc

```
Processing /dist/gateau-0.2.2.tar.gz
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
    Preparing wheel metadata ... error
    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmpt0g2yj3m_in_process.py prepare_metadata_for_build_whe
el /tmp/tmp3mfvm0xi
         cwd: /tmp/pip-req-build-e80w_flj
    Complete output (20 lines):
    + meson setup /tmp/pip-req-build-e80w_flj /tmp/pip-req-build-e80w_flj/.mesonpy-ksppikbp -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-e80w_flj/.
mesonpy-ksppikbp/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-e80w_flj
    Build dir: /tmp/pip-req-build-e80w_flj/.mesonpy-ksppikbp
    Build type: native build
    Project name: gateau
    Project version: 0.2.2
    
    ../meson.build:1:0: ERROR: Unknown compiler(s): [['cc'], ['gcc'], ['clang'], ['nvc'], ['pgc
c'], ['icc'], ['icx']]
    The following exception(s) were encountered:
    Running `cc --version` gave "[Errno 2] No such file or directory: 'cc'"
    Running `gcc --version` gave "[Errno 2] No such file or directory: 'gcc'"
    Running `clang --version` gave "[Errno 2] No such file or directory: 'clang'"
    Running `nvc --version` gave "[Errno 2] No such file or directory: 'nvc'"
    Running `pgcc --version` gave "[Errno 2] No such file or directory: 'pgcc'"
    Running `icc --version` gave "[Errno 2] No such file or directory: 'icc'"
    Running `icx --version` gave "[Errno 2] No such file or directory: 'icx'"
    
    A full log can be found at /tmp/pip-req-build-e80w_flj/.mesonpy-ksppikbp/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmpt0g2yj3m_in_process.
py prepare_metadata_for_build_wheel /tmp/tmp3mfvm0xi Check the logs for full command output.
/venv/bin/python: No module named gateau
```

no pkg-config

```

+ docker run --rm -ti -v ./dist:/dist:ro gateau-ubuntu-20-04 sh -c '. /venv/bin/activate; pip i
nstall /dist/*.tar.gz ; python -m gateau'
Processing /dist/gateau-0.2.2.tar.gz
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
    Preparing wheel metadata ... error
    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmppw6sqwru_in_process.py prepare_metadata_for_build_whe
el /tmp/tmp6v7ncexp
         cwd: /tmp/pip-req-build-z5nlf0xq
    Complete output (30 lines):
    + meson setup /tmp/pip-req-build-z5nlf0xq /tmp/pip-req-build-z5nlf0xq/.mesonpy-2uqnmplq -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-z5nlf0xq/.
mesonpy-2uqnmplq/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-z5nlf0xq
    Build dir: /tmp/pip-req-build-z5nlf0xq/.mesonpy-2uqnmplq
    Build type: native build
    Project name: gateau
    Project version: 0.2.2
    C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
    C linker for the host machine: cc ld.bfd 2.34
    C++ compiler for the host machine: c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.
0")
    C++ linker for the host machine: c++ ld.bfd 2.34
    Cuda compiler for the host machine: nvcc (nvcc 12.3.107
    Build cuda_12.3.r12.3/compiler.33567101_0)
    Cuda linker for the host machine: nvcc nvlink 12.3.107
    Build cuda_12.3.r12.3/compiler.33567101_0
    Host machine cpu family: x86_64
    Host machine cpu: x86_64
    Program python found: YES (/venv/bin/python3.9)
    Run-time dependency CUDA (modules: cudart_static, cufft) found: YES 12.3 (/usr/local/cuda)
    Message: Meson could find cuda libraries normally, yay!
    Did not find pkg-config by name 'pkg-config'
    Found pkg-config: NO
    Did not find CMake 'cmake'
    Found CMake: NO
    Run-time dependency gsl found: NO
    
    ../meson.build:111:6: ERROR: Dependency lookup for gsl with method 'pkgconfig' failed: Pkg-
config for machine host machine not found. Giving up.
    
    A full log can be found at /tmp/pip-req-build-z5nlf0xq/.mesonpy-2uqnmplq/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmppw6sqwru_in_process.
py prepare_metadata_for_build_wheel /tmp/tmp6v7ncexp Check the logs for full command output.
/venv/bin/python: No module named gateau
```

---


No gcc

```

root@50ddd7cba984:/app# 

root@50ddd7cba984:/app# ls
requirements.noextras.txt
root@50ddd7cba984:/app# . /venv/bin/activate
(venv) root@50ddd7cba984:/app# pip install ^C  
(venv) root@50ddd7cba984:/app# ll di
ls: cannot access 'di': No such file or directory
(venv) root@50ddd7cba984:/app# ll di^C
(venv) root@50ddd7cba984:/app# ls /dist/gateau-0.2.2.tar.gz 
/dist/gateau-0.2.2.tar.gz
(venv) root@50ddd7cba984:/app# pip install /dist/gateau-0.2.2.tar.gz 
Processing /dist/gateau-0.2.2.tar.gz
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
    Preparing wheel metadata ... error
    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmpuun301jn_in_process.py prepare_metadata_for_build_whe
el /tmp/tmp5rnf3tbt
         cwd: /tmp/pip-req-build-z43evrp4
    Complete output (44 lines):
    + meson setup /tmp/pip-req-build-z43evrp4 /tmp/pip-req-build-z43evrp4/.mesonpy-7qkicf2b -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-z43evrp4/.
mesonpy-7qkicf2b/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-z43evrp4
    Build dir: /tmp/pip-req-build-z43evrp4/.mesonpy-7qkicf2b
    Build type: native build
    ../meson.build:26: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
      error('
            ^
    ../meson.build:32: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
      error('
            ^
    ../meson.build:54: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
      error('
            ^
    ../meson.build:58: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
      error('
            ^
    ../meson.build:62: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
      error('
            ^
    ../meson.build:134: WARNING: Newline character in a string detected, use ''' (three single 
quotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
        error('
              ^
    Project name: gateau
    Project version: 0.2.2
    
    ../meson.build:1:0: ERROR: Unknown compiler(s): [['cc'], ['gcc'], ['clang'], ['nvc'], ['pgc
c'], ['icc'], ['icx']]
    The following exception(s) were encountered:
    Running `cc --version` gave "[Errno 2] No such file or directory: 'cc'"
    Running `gcc --version` gave "[Errno 2] No such file or directory: 'gcc'"
    Running `clang --version` gave "[Errno 2] No such file or directory: 'clang'"
    Running `nvc --version` gave "[Errno 2] No such file or directory: 'nvc'"
    Running `pgcc --version` gave "[Errno 2] No such file or directory: 'pgcc'"
    Running `icc --version` gave "[Errno 2] No such file or directory: 'icc'"
    Running `icx --version` gave "[Errno 2] No such file or directory: 'icx'"
    
    A full log can be found at /tmp/pip-req-build-z43evrp4/.mesonpy-7qkicf2b/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmpuun301jn_in_process.
py prepare_metadata_for_build_wheel /tmp/tmp5rnf3tbt Check the logs for full command output.
```

- apt install gcc

## No g++

```

    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmp6rw8pti5_in_process.py prepare_metadata_for_build_whe
el /tmp/tmpyb7vtvi5
         cwd: /tmp/pip-req-build-01z0zrtn
    Complete output (26 lines):
    + meson setup /tmp/pip-req-build-01z0zrtn /tmp/pip-req-build-01z0zrtn/.mesonpy-vuoj6q3e -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-01z0zrtn/.
mesonpy-vuoj6q3e/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-01z0zrtn
    Build dir: /tmp/pip-req-build-01z0zrtn/.mesonpy-vuoj6q3e
    Build type: native build
    ../meson.build:96: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
        error('
              ^
    Project name: gateau
    Project version: 0.2.2
    C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
    C linker for the host machine: cc ld.bfd 2.34
    
    ../meson.build:1:0: ERROR: Unknown compiler(s): [['c++'], ['g++'], ['clang++'], ['nvc++'], 
['pgc++'], ['icpc'], ['icpx']]
    The following exception(s) were encountered:
    Running `c++ --version` gave "[Errno 2] No such file or directory: 'c++'"
    Running `g++ --version` gave "[Errno 2] No such file or directory: 'g++'"
    Running `clang++ --version` gave "[Errno 2] No such file or directory: 'clang++'"
    Running `nvc++ --version` gave "[Errno 2] No such file or directory: 'nvc++'"
    Running `pgc++ --version` gave "[Errno 2] No such file or directory: 'pgc++'"
    Running `icpc --version` gave "[Errno 2] No such file or directory: 'icpc'"
    Running `icpx --version` gave "[Errno 2] No such file or directory: 'icpx'"
    
    A full log can be found at /tmp/pip-req-build-01z0zrtn/.mesonpy-vuoj6q3e/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmp6rw8pti5_in_process.
py prepare_metadata_for_build_wheel /tmp/tmpyb7vtvi5 Check the logs for full command output.

```

- apt install g++

## No cuda


```
    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmpms1dic6f_in_process.py prepare_metadata_for_build_whe
el /tmp/tmpaf8ly2ms
         cwd: /tmp/pip-req-build-doidj2xf
    Complete output (20 lines):
    + meson setup /tmp/pip-req-build-doidj2xf /tmp/pip-req-build-doidj2xf/.mesonpy-92gnl3m3 -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-doidj2xf/.
mesonpy-92gnl3m3/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-doidj2xf
    Build dir: /tmp/pip-req-build-doidj2xf/.mesonpy-92gnl3m3
    Build type: native build
    ../meson.build:96: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
        error('
              ^
    Project name: gateau
    Project version: 0.2.2
    C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
    C linker for the host machine: cc ld.bfd 2.34
    C++ compiler for the host machine: c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.
0")
    C++ linker for the host machine: c++ ld.bfd 2.34
    
    ../meson.build:1:0: ERROR: Could not find suitable CUDA compiler: "nvcc"
    
    A full log can be found at /tmp/pip-req-build-doidj2xf/.mesonpy-92gnl3m3/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmpms1dic6f_in_process.
py prepare_metadata_for_build_wheel /tmp/tmpaf8ly2ms Check the logs for full command output.
```

- follow instructions

## No gsl

```
    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmpt7ybi6tk_in_process.py prepare_metadata_for_build_whe
el /tmp/tmpmt3na83q
         cwd: /tmp/pip-req-build-7ccw5s1g
    Complete output (32 lines):
    + meson setup /tmp/pip-req-build-7ccw5s1g /tmp/pip-req-build-7ccw5s1g/.mesonpy-6rdlqraj -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-7ccw5s1g/.
mesonpy-6rdlqraj/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-7ccw5s1g
    Build dir: /tmp/pip-req-build-7ccw5s1g/.mesonpy-6rdlqraj
    Build type: native build
    ../meson.build:96: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
        error('
              ^
    Project name: gateau
    Project version: 0.2.2
    C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
    C linker for the host machine: cc ld.bfd 2.34
    C++ compiler for the host machine: c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.
0")
    C++ linker for the host machine: c++ ld.bfd 2.34
    Cuda compiler for the host machine: nvcc (nvcc 12.3.107
    Build cuda_12.3.r12.3/compiler.33567101_0)
    Cuda linker for the host machine: nvcc nvlink 12.3.107
    Build cuda_12.3.r12.3/compiler.33567101_0
    Host machine cpu family: x86_64
    Host machine cpu: x86_64
    Program python found: YES (/venv/bin/python3.9)
    Did not find pkg-config by name 'pkg-config'
    Found pkg-config: NO
    Did not find CMake 'cmake'
    Found CMake: NO
    Run-time dependency gsl found: NO
    
    ../meson.build:16:6: ERROR: Dependency lookup for gsl with method 'pkgconfig' failed: Pkg-c
onfig for machine host machine not found. Giving up.
    
    A full log can be found at /tmp/pip-req-build-7ccw5s1g/.mesonpy-6rdlqraj/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmpt7ybi6tk_in_process.
py prepare_metadata_for_build_wheel /tmp/tmpmt3na83q Check the logs for full command output.
```

- no libgsl AND/OR no pkg-config
- sudo apt install pkg-config libgsl-dev

## No hdf5


```
Processing /dist/gateau-0.2.2.tar.gz
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
    Preparing wheel metadata ... error
    ERROR: Command errored out with exit status 1:
     command: /venv/bin/python3.9 /tmp/tmpwm6rp63x_in_process.py prepare_metadata_for_build_whe
el /tmp/tmpws_ip8ze
         cwd: /tmp/pip-req-build-eu78x4os
    Complete output (33 lines):
    + meson setup /tmp/pip-req-build-eu78x4os /tmp/pip-req-build-eu78x4os/.mesonpy-hvreehqw -Db
uildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-req-build-eu78x4os/.
mesonpy-hvreehqw/meson-python-native-file.ini
    The Meson build system
    Version: 1.10.1
    Source dir: /tmp/pip-req-build-eu78x4os
    Build dir: /tmp/pip-req-build-eu78x4os/.mesonpy-hvreehqw
    Build type: native build
    ../meson.build:96: WARNING: Newline character in a string detected, use ''' (three single q
uotes) for multiline strings instead.
    This will become a hard error in a future Meson release.
        error('
              ^
    Project name: gateau
    Project version: 0.2.2
    C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
    C linker for the host machine: cc ld.bfd 2.34
    C++ compiler for the host machine: c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.
0")
    C++ linker for the host machine: c++ ld.bfd 2.34
    Cuda compiler for the host machine: nvcc (nvcc 12.3.107
    Build cuda_12.3.r12.3/compiler.33567101_0)
    Cuda linker for the host machine: nvcc nvlink 12.3.107
    Build cuda_12.3.r12.3/compiler.33567101_0
    Host machine cpu family: x86_64
    Host machine cpu: x86_64
    Program python found: YES (/venv/bin/python3.9)
    Found pkg-config: YES (/usr/bin/pkg-config) 0.29.1
    Run-time dependency gsl found: YES 2.5
    Library gslcblas found: YES
    Found pkg-config: YES (/usr/bin/pkg-config) 0.29.1
    h5cc found: NO
    Run-time dependency HDF5 found: NO (tried pkgconfig, pkgconfig and config-tool)
    
    ../meson.build:18:7: ERROR: Dependency "hdf5" not found, tried pkgconfig, pkgconfig and con
fig-tool
    
    A full log can be found at /tmp/pip-req-build-eu78x4os/.mesonpy-hvreehqw/meson-logs/meson-l
og.txt
    ----------------------------------------
ERROR: Command errored out with exit status 1: /venv/bin/python3.9 /tmp/tmpwm6rp63x_in_process.
py prepare_metadata_for_build_wheel /tmp/tmpws_ip8ze Check the logs for full command output.

```

- apt install libhdf5-dev

