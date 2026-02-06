@echo off
setlocal

ECHO Switching to shared folder...
Z:

if not defined GATEAU_SKIP_DEPS (
	ECHO Installing deps...
	ECHO set "GATEAU_SKIP_DEPS=1" to skip this.

	C:\Users\maybetree\AppData\Local\Programs\Python\Python312\python.exe -m pip install --no-index --find-links=win\tmp\pipcache -r win\requirements.txt
	C:\Users\maybetree\AppData\Local\Programs\Python\Python312\python.exe -m pip install --no-index --find-links=win\tmp\pipcache build 
) else (
	ECHO Skip installing deps.
)

ECHO Building...

rmdir /s /q build.win

C:\Users\maybetree\AppData\Local\Programs\Python\Python312\python.exe -m build --verbose --verbose --verbose --wheel --no-isolation --config-setting=builddir=build.win

REM --wheel is needed because... no clue honestly?
REM But doesn't work without it
REM see <https://github.com/mesonbuild/meson-python/issues/507>
REM
REM --no-isolation tells build to use the pkgs we just installed
REM instead of trying to dl them from the internet
REM
REM --config-settings=builddir=build.win
REM tells mesonpy to use a persistent
REM `build.win` instead of a temporary directory
REM for build files.
REM This is required because otherwise mesonpy would crash trying to clean
REM up the temporary directory due to windows file locking stuff.

REM ECHO Renaming wheel...
REM 
REM setlocal enabledelayedexpansion
REM 
REM REM Thank you chatgpt for this one
REM for %%f in (dist\gateau-*-cp*-cp*-win_amd64.whl) do (
REM     set "fname=%%~nxf"
REM     for /f "tokens=2 delims=-" %%v in ("!fname!") do (
REM         ren "%%f" "gateau-%%v-cp39.cp310.cp311.cp312.cp313-none-win_amd64.whl"
REM     )
REM )
REM 
REM dir dist

ECHO Build script finished.

