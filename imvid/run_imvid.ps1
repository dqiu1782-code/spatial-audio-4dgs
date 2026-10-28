# ImViD Opera on Windows, after imvid_prepare / imvid_track / imvid_finalize / imvid_audio (run anywhere):
# PNG + COLMAP init cloud -> 4DGaussians (deformation mask) -> render the held-out camera -> PSNR.
#   powershell -File run_imvid.ps1 -Data E:\4dgs_data\imvid_opera -GS E:\4DGaussians-master -Py python
param([string]$Data = 'E:\4dgs_data\imvid_opera', [string]$GS = 'E:\4DGaussians-master', [string]$Py = 'python')
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $Data 'run.log'
function L($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File -Append -Encoding utf8 $log }
function Run($cmd) { L "RUN $cmd"; cmd /c "$cmd >> `"$log`" 2>&1"; if ($LASTEXITCODE -ne 0) { L "FAILED ($LASTEXITCODE)"; exit 1 } }
L 'start'
Run "$Py -u `"$here\imvid_colmap.py`" $Data"
Set-Content -Path "$GS\arguments\dnerf\imvid.py" -Value "_base_ = './replay.py'`nOptimizationParams = dict(`n    iterations=14000,`n)" -Encoding ascii
Set-Location $GS
$env:BD_DYN_MASK = "$Data\dyn_mask.npz"
Run "$Py -u train.py -s $Data --port 6021 --expname imvid_opera --configs arguments\dnerf\imvid.py --sh_degree 1"
Run "$Py -u render.py --model_path output\imvid_opera --skip_train --skip_video --configs arguments\dnerf\imvid.py --sh_degree 1"
Run "$Py -u `"$here\psnr_dir.py`" output\imvid_opera"
L 'IMVID_ALL_DONE'
