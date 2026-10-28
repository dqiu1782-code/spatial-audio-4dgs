@echo off
REM run_demo.bat - simulate 4 cameras with microphones, localize the sound source, render binaural audio
cd /d %~dp0
set PY=E:\lkenv\Scripts\python.exe
if not exist "%PY%" set PY=python
echo === 1. four cameras, 4-mic arrays, spread around the room
%PY% simulate.py --out demo_tetra --array tetra --cams 4 || goto :err
%PY% run_localize.py demo_tetra || goto :err
%PY% binaural.py demo_tetra || goto :err
%PY% binaural.py demo_tetra --use truth || goto :err
%PY% binaural.py demo_tetra --listener cam1 || goto :err
echo === 2. same, but only 2 mics per camera (stereo)
%PY% simulate.py --out demo_stereo --array stereo --cams 4 || goto :err
%PY% run_localize.py demo_stereo || goto :err
echo === 3. cameras clustered together
%PY% simulate.py --out demo_tight --array tetra --spread tight --cams 4 || goto :err
%PY% run_localize.py demo_tight || goto :err
echo DEMO_DONE
goto :eof
:err
echo DEMO_FAILED
