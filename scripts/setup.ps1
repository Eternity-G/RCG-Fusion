$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$bootstrapPython = Join-Path (Split-Path -Parent $projectRoot) '.bootstrap\Scripts\python.exe'
$uvTool = Join-Path (Split-Path -Parent $projectRoot) '.bootstrap\Scripts\uv.exe'
if (-not (Test-Path -LiteralPath $uvTool)) {
    python -m venv (Join-Path (Split-Path -Parent $projectRoot) '.bootstrap')
    if ($LASTEXITCODE) { throw 'Bootstrap Python creation failed' }
    & $bootstrapPython -m pip install uv
    if ($LASTEXITCODE) { throw 'uv installation failed' }
}
& $uvTool python install 3.11
if ($LASTEXITCODE) { throw 'Python installation failed' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $uvTool venv --python 3.11 .venv
    if ($LASTEXITCODE) { throw 'Virtual environment creation failed' }
}
# This wheel is compatible with the inspected NVIDIA 560.81 driver. The helper
# validates the official PyTorch index hash and resumes interrupted downloads.
& $uvTool pip install --python .venv/Scripts/python.exe numpy==1.26.4
if ($LASTEXITCODE) { throw 'Bootstrap NumPy installation failed' }
& .\.venv\Scripts\python.exe scripts/fetch_torch.py
if ($LASTEXITCODE) { throw 'Verified CUDA wheel download failed' }
& $uvTool pip install --python .venv/Scripts/python.exe '.wheels/torch-2.6.0+cu124-cp311-cp311-win_amd64.whl'
if ($LASTEXITCODE) { throw 'CUDA wheel install failed' }
& $uvTool pip install --python .venv/Scripts/python.exe torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE) { throw 'TorchVision/TorchAudio CUDA wheel install failed' }
& $uvTool pip install --python .venv/Scripts/python.exe -e .
if ($LASTEXITCODE) { throw 'Project installation failed' }
& .\.venv\Scripts\python.exe -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
if ($LASTEXITCODE) { throw 'CUDA verification failed' }
