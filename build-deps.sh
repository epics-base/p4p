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
git clone --quiet --depth 5 --branch "$BRPVD" https://github.com/epics-base/pvDataCPP.git pvDataCPP
git clone --quiet --depth 5 --branch "$BRPVA" https://github.com/epics-base/pvAccessCPP.git pvAccessCPP

cat << EOF > pvDataCPP/configure/RELEASE.local
EPICS_BASE=$HOME/.source/epics-base
EOF

cat << EOF > pvAccessCPP/configure/RELEASE.local
EPICS_BASE=$HOME/.source/epics-base
PVDATA=$HOME/.source/pvDataCPP
EOF

make -j2 -C epics-base
make -j2 -C pvDataCPP
make -j2 -C pvAccessCPP
