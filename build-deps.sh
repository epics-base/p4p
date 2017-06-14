#!/bin/sh
set -e -x

CURDIR="$PWD"

cat << EOF > configure/RELEASE.local
EPICS_BASE=$HOME/.source/epics-base
PVDATA=$HOME/.source/pvDataCPP
PVACCESS=$HOME/.source/pvAccessCPP
EOF
cat configure/RELEASE.local

install -d "$HOME/.source"
cd "$HOME/.source"

git clone --quiet --depth 5 --branch "$BRBASE" https://github.com/epics-base/epics-base.git epics-base
git clone --quiet --depth 5 --branch "$BRPVD" https://github.com/mdavidsaver/pvDataCPP.git pvDataCPP
git clone --quiet --depth 5 --branch "$BRPVA" https://github.com/mdavidsaver/pvAccessCPP.git pvAccessCPP

(cd epics-base && git log -n1 )
(cd pvDataCPP && git log -n1 )
(cd pvAccessCPP && git log -n1 )

EPICS_HOST_ARCH=`sh epics-base/startup/EpicsHostArch`

case "$CMPLR" in
clang)
  echo "Host compiler is clang"
  cat << EOF >> epics-base/configure/os/CONFIG_SITE.Common.$EPICS_HOST_ARCH
GNU         = NO
CMPLR_CLASS = clang
CC          = clang
CCC         = clang++
EOF

  # hack
  sed -i -e 's/CMPLR_CLASS = gcc/CMPLR_CLASS = clang/' epics-base/configure/CONFIG.gnuCommon

  clang --version
  ;;
*)
  echo "Host compiler is default"
  gcc --version
  ;;
esac

cat << EOF > pvDataCPP/configure/RELEASE.local
EPICS_BASE=$HOME/.source/epics-base
EOF

cat << EOF > pvAccessCPP/configure/RELEASE.local
PVDATA=$HOME/.source/pvDataCPP
EPICS_BASE=$HOME/.source/epics-base
EOF

make -j2 -C epics-base
make -j2 -C pvDataCPP
make -j2 -C pvAccessCPP
