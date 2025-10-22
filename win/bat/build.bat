@echo off
setlocal

ECHO Switching to shared folder...
Z:

ECHO Installing deps...

pip install --no-index --find-links=win\pipcache -r win\requirements.txt
pip install --no-index --find-links=win\pipcache build

ECHO Building...

python -m build --no-isolation

REM --no-isolation tells build to use the pkgs we just installed
REM instead of trying to dl them from the internet

ECHO Renaming wheel...

setlocal enabledelayedexpansion

for %%f in (dist\gateau-*-cp*-cp*-win_amd64.whl) do (
    set "fname=%%~nxf"
    for /f "tokens=2 delims=-" %%v in ("!fname!") do (
        ren "%%f" "gateau-%%v-cp39.cp310.cp311.cp312.cp313-none-win_amd64.whl"
    )
)
REM Thank you chatgpt for this one

dir dist

ECHO Build script finished.

