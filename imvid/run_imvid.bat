@echo off
rem Braindance viewer on the ImViD Opera scene: 4DGS picture + spatial audio from the 39 recordings.
rem Edit the three paths below for your machine.
set DATA=E:\4dgs_data\imvid_opera
set GS=E:\4DGaussians-master
set KEMAR=E:\replay\kemar
set BD_DYN_MASK=%DATA%\dyn_mask.npz
cd /d %GS%
python "%~dp0..\viewer\braindance_viewer.py" --model output\imvid_opera --configs arguments\dnerf\imvid.py --sound %DATA%\sound --data %DATA% --hrtf %KEMAR% --reverb 0.08 %*
pause
