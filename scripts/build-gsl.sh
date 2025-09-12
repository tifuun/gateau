#!/bin/sh

# Ubuntu deps:
# wget autotools-dev autoconf libtool

set -e
#rm -rf gsl
mkdir -p gsl
cd gsl

#wget https://mirror.clientvps.com/gnu/gsl/gsl-2.5.tar.gz
#wget https://mirror.clientvps.com/gnu/gsl/gsl-2.5.tar.gz.sig
#wget https://www.gnu.org/software/gsl/key/gsl_key.txt

gpg --import gsl_key.txt
gpg --verify gsl-2.5.tar.gz.sig

tar xf gsl-2.5.tar.gz
cd gsl-2.5

./autogen.sh

# the ./configure file expects /bin/sh to be bash-compatible
# apparently, but on ubuntu bin/sh is dash.
# Fix by overriding shebang.
sed -i -e '1i #!/bin/bash' ./configure

./configure --enable-maintainer-mode
make -j$(nproc)
make install

