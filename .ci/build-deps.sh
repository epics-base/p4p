#!/bin/sh
set -e -x

CURDIR="$PWD"

install -d "$HOME/.source"
cd "$HOME/.source"

if [ "$BRPVA" ]
then
    git clone --quiet --depth 5 --branch "$BRPVA" https://github.com/epics-base/pvAccessCPP.git pvAccessCPP
    (cd pvAccessCPP && git log -n1 )
    cat << EOF >> $CURDIR/configure/RELEASE.local
PVACCESS=$HOME/.source/pvAccessCPP
EOF
    cat << EOF > pvAccessCPP/configure/RELEASE.local
PVDATA=$HOME/.source/pvDataCPP
EPICS_BASE=$HOME/.source/epics-base
EOF
fi

if [ "$BRPVD" ]
then
    git clone --quiet --depth 5 --branch "$BRPVD" https://github.com/epics-base/pvDataCPP.git pvDataCPP
    (cd pvDataCPP && git log -n1 )
    cat << EOF >> $CURDIR/configure/RELEASE.local
PVDATA=$HOME/.source/pvDataCPP
EOF
    cat << EOF > pvDataCPP/configure/RELEASE.local
EPICS_BASE=$HOME/.source/epics-base
EOF
fi

git clone --quiet --depth 5 --branch "$BRBASE" https://github.com/epics-base/epics-base.git epics-base
(cd epics-base && git log -n1 )

cat << EOF >> $CURDIR/configure/RELEASE.local
EPICS_BASE=$HOME/.source/epics-base
EOF
cat $CURDIR/configure/RELEASE.local

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

make -j2 -C epics-base "$@"
[ "$BRPVD" ] && make -j2 -C pvDataCPP "$@"
[ "$BRPVA" ] && make -j2 -C pvAccessCPP "$@"
