
git clone --quiet --depth 5 --branch "%BRBASE%" https://github.com/epics-base/epics-base.git epics-base
git clone --quiet --depth 5 --branch "%BRPVD%" https://github.com/epics-base/pvDataCPP.git pvDataCPP
git clone --quiet --depth 5 --branch "%BRPVA%" https://github.com/epics-base/pvAccessCPP.git pvAccessCPP

echo EPICS_BASE=%CD%/epics-base > pvDataCPP\configure\RELEASE.local

echo EPICS_BASE=%CD%/epics-base > pvAccessCPP\configure\RELEASE.local
echo PVDATA=%CD%/pvDataCPP >> pvAccessCPP\configure\RELEASE.local

echo EPICS_BASE=%CD%/epics-base > configure\RELEASE.local
echo PVDATA=%CD%/pvDataCPP >> configure\RELEASE.local
echo PVACCESS=%CD%/pvAccessCPP >> configure\RELEASE.local

set BASEDIR=%CD%

echo [INFO] Installing Make 4.1
curl -fsS --retry 3 -o C:\tools\make-4.1.zip https://epics.anl.gov/download/tools/make-4.1-win64.zip
cd \tools
"C:\Program Files\7-Zip\7z" e make-4.1.zip

cd %BASEDIR%

set TOOLCHAIN=2017

set "VSINSTALL=C:\Program Files (x86)\Microsoft Visual Studio %TOOLCHAIN%"
if not exist "%VSINSTALL%\" set "VSINSTALL=C:\Program Files (x86)\Microsoft Visual Studio\%TOOLCHAIN%\Community"
if not exist "%VSINSTALL%\" goto MSMissing

set "MAKE=C:\tools\make"


set EPICS_HOST_ARCH=windows-x64
:: VS 2017
if exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" (
    call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
    where cl
    if !ERRORLEVEL! NEQ 0 goto MSMissing
    goto MSFound
)
if exist "%VSINSTALL%\VC\vcvarsall.bat" (
    call "%VSINSTALL%\VC\vcvarsall.bat" amd64
    where cl
    if !ERRORLEVEL! NEQ 0 (
        call "%VSINSTALL%\VC\vcvarsall.bat" x86_amd64
        where cl
        if !ERRORLEVEL! NEQ 0 goto MSMissing
    )
    goto MSFound
)
if exist "%VSINSTALL%\VC\bin\amd64\vcvars64.bat" (
    call "%VSINSTALL%\VC\bin\amd64\vcvars64.bat"
    where cl
    if !ERRORLEVEL! NEQ 0 goto MSMissing
    goto MSFound
)


:MSMissing
echo [INFO] Installation for MSVC Toolchain %TOOLCHAIN% / %OS% seems to be missing
exit 1

:MSFound
echo [INFO] Microsoft Visual Studio Toolchain %TOOLCHAIN%
echo [INFO] Compiler Version
cl

:Finish
echo [INFO] EPICS_HOST_ARCH: %EPICS_HOST_ARCH%
echo [INFO] Make version
%MAKE% --version
echo [INFO] Perl version
perl --version

%MAKE% -j2 -C epics-base %*
%MAKE% -j2 -C pvDataCPP %*
%MAKE% -j2 -C pvAccessCPP %*
