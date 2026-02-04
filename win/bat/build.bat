@echo off
setlocal

ECHO Switching to shared folder...
Z:

ECHO Installing deps...

C:\Users\maybetree\AppData\Local\Programs\Python\Python312\python.exe -m pip install --no-index --find-links=win\pipcache -r win\requirements.txt
C:\Users\maybetree\AppData\Local\Programs\Python\Python312\python.exe -m pip install --no-index --find-links=win\pipcache build

ECHO Building...

C:\Users\maybetree\AppData\Local\Programs\Python\Python312\python.exe -m build --no-isolation

REM --no-isolation tells build to use the pkgs we just installed
REM instead of trying to dl them from the internet

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

