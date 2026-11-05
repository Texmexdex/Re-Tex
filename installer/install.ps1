# Re-Tex installer for Windows 10/11 with an NVIDIA GPU. Started by install.bat; safe to run again (it skips
# what's done). Everything goes in this folder except Python, FFmpeg and Node (winget, for this user) and the
# Hugging Face download cache (your user folder). ASCII only: Windows PowerShell 5.1 reads this file.
param(
    [ValidateSet("", "Full", "Core")] [string]$Mode = "",
    [switch]$NoModels,          # set up the software only (for testing)
    [switch]$NoShortcut,        # don't touch the desktop (for testing)
    [switch]$Yes                # don't ask; take the defaults
)
$ErrorActionPreference = "Continue"     # native tools write progress to stderr; failures are checked by exit code
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv-phase2\Scripts\python.exe"
$PyL = Join-Path $Root ".venv-listen\Scripts\python.exe"
$Problems = New-Object System.Collections.Generic.List[string]

function Say($t, $c = "Gray") { Write-Host $t -ForegroundColor $c }
function Step($t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }
function Ask($q, $default = "y") {
    if ($Yes) { return ($default -eq "y") }
    $a = Read-Host "$q [$(if ($default -eq 'y') {'Y/n'} else {'y/N'})]"
    if ($a -eq "") { $a = $default }
    return ($a -match "^[yY]")
}
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Run($exe, [string[]]$a) {
    & $exe @a
    if ($LASTEXITCODE -ne 0) { throw "$exe $($a -join ' ') failed (exit $LASTEXITCODE)" }
}
function Winget-Install($id, $what) {
    if (-not (Have "winget")) {
        throw "winget isn't available, so $what can't be installed automatically. Install 'App Installer' from the Microsoft Store (it provides winget), or install $what yourself, then run install.bat again."
    }
    Say "  installing $what with winget ($id)..."
    & winget install -e --id $id --scope user --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        # some packages only install machine-wide
        & winget install -e --id $id --silent --accept-package-agreements --accept-source-agreements
    }
    Refresh-Path
}

Say ""
Say "  Re-Tex installer" "White"
Say "  Folder: $Root"

# ------------------------------------------------------------------ what to install
if ($Mode -eq "") {
    Say ""
    Say "  1) Full - everything: the app, all separation / beat / lyrics models, MuScriptor, Gemma (offline DJ"
    Say "     planner) and Music Flamingo (song descriptions). About 50 GB."
    Say "  2) Core - the app and the models it needs to work. About 15 GB."
    $Mode = "Full"
    if (-not $Yes) {
        $a = Read-Host "  Which one? [1]"
        if ($a -eq "2") { $Mode = "Core" }
    }
}
Say "  Installing: $Mode" "White"
$need = if ($Mode -eq "Full") { 55 } else { 18 }
$free = [math]::Round((Get-PSDrive ($Root.Substring(0, 1))).Free / 1GB)
if ($free -lt $need) {
    Say "  Only $free GB free on this drive; about $need GB is needed." "Yellow"
    if (-not (Ask "  Continue anyway?" "n")) { exit 1 }
}

# ------------------------------------------------------------------ GPU
Step "GPU"
if (Have "nvidia-smi") {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    Say "  $gpu" "Green"
} else {
    Say "  No NVIDIA GPU found (nvidia-smi isn't on this PC). Re-Tex will still run, but separating songs and" "Yellow"
    Say "  transcribing lyrics will be much slower, and Music Flamingo won't run." "Yellow"
    if (-not (Ask "  Continue?" "y")) { exit 1 }
}

# ------------------------------------------------------------------ Python, FFmpeg, Node
Step "Python 3.13, FFmpeg, Node.js"
function Find-Python313 {
    if (Have "py") {
        $p = (& py -3.13 -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    }
    $c = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
    if (Test-Path $c) { return $c }
    return $null
}
$Base = Find-Python313
if (-not $Base) { Winget-Install "Python.Python.3.13" "Python 3.13"; $Base = Find-Python313 }
if (-not $Base) { throw "Python 3.13 couldn't be found after installing it. Install it from https://www.python.org/downloads/ and run install.bat again." }
Say "  Python: $Base" "Green"
if (-not (Have "ffmpeg")) { Winget-Install "Gyan.FFmpeg" "FFmpeg" }
if (Have "ffmpeg") { Say "  FFmpeg: ok" "Green" } else { $Problems.Add("FFmpeg isn't on PATH yet (open a new window, or install it from https://www.gyan.dev/ffmpeg/builds/). Without it mp3/m4a can't be read or exported.") }
if (-not (Have "node")) { Winget-Install "OpenJS.NodeJS.LTS" "Node.js" }
if (Have "node") { Say "  Node.js: $(& node -v)" "Green" } else { $Problems.Add("Node.js isn't on PATH yet; ReadHead needs it (https://nodejs.org).") }

# ------------------------------------------------------------------ the app's environment
Step "Re-Tex's Python environment (.venv-phase2)"
if (Test-Path $Py) {
    Say "  already there; updating it to the pinned versions"
} else {
    Run $Base @("-m", "venv", ".venv-phase2")
}
Run $Py @("-m", "pip", "install", "--upgrade", "pip", "--quiet", "--disable-pip-version-check")
Say "  PyTorch 2.8.0 for CUDA 12.8 (about 3 GB the first time)..."
Run $Py @("-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128", "--disable-pip-version-check")
Say "  everything else..."
Run $Py @("-m", "pip", "install", "-r", "installer\requirements-core.txt", "--disable-pip-version-check")
$cuda = (& $Py -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null)
if ($cuda -eq "cuda") { Say "  PyTorch sees the GPU" "Green" } else { Say "  PyTorch is running on the CPU" "Yellow" }

# ------------------------------------------------------------------ models
if (-not $NoModels) {
    Step "Models (separation, beats, lyrics, word timing)"
    & $Py "installer\fetch_models.py" core
    if ($LASTEXITCODE -ne 0) { $Problems.Add("Some core models didn't download (see above). Run install.bat again to retry.") }
}

# ------------------------------------------------------------------ Full: extras
if ($Mode -eq "Full") {
    Step "Extras: MuScriptor and llama.cpp"
    try {
        Run $Py @("-m", "pip", "install", "muscriptor==0.3.0", "--disable-pip-version-check")
    } catch { $Problems.Add("MuScriptor's package didn't install: $_") }
    $llama = $false
    try {
        # a prebuilt CUDA build of llama-cpp-python; without one, pip would need a C++ compiler
        Run $Py @("-m", "pip", "install", "llama-cpp-python", "--only-binary", ":all:", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124", "--disable-pip-version-check")
        $llama = $true
    } catch {
        $Problems.Add("llama-cpp-python has no ready-made build for this Python yet, so the offline Gemma planner stays off. DJ Mix still plans with Claude (if Claude Code is set up) or its rules.")
    }

    if (-not $NoModels) {
        Step "Hugging Face login (for MuScriptor and Gemma)"
        $logged = (& $Py -c "from huggingface_hub import get_token; print('yes' if get_token() else 'no')" 2>$null)
        if ($logged -ne "yes") {
            Say "  MuScriptor and Gemma are free but gated: their authors ask you to log in and accept their terms."
            Say "  You need a free Hugging Face account and an access token (type 'Read')."
            if (Ask "  Log in now? (opens the token page in your browser)" "y") {
                Start-Process "https://huggingface.co/settings/tokens"
                & (Join-Path $Root ".venv-phase2\Scripts\hf.exe") auth login
                $logged = (& $Py -c "from huggingface_hub import get_token; print('yes' if get_token() else 'no')" 2>$null)
            }
        }
        if ($logged -eq "yes") {
            Say "  Accept the terms on these two pages (one click each), then come back here:"
            Say "    https://huggingface.co/MuScriptor/muscriptor-large"
            Say "    https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf"
            if (-not $Yes) {
                Start-Process "https://huggingface.co/MuScriptor/muscriptor-large"
                Start-Process "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf"
                Read-Host "  Press Enter when you've accepted both" | Out-Null
            }
            & $Py "installer\fetch_models.py" muscriptor
            if ($LASTEXITCODE -ne 0) { $Problems.Add("MuScriptor's model didn't download (terms not accepted yet?). Run install.bat again later.") }
            if ($llama) {
                & $Py "installer\fetch_models.py" gemma
                if ($LASTEXITCODE -ne 0) { $Problems.Add("Gemma didn't download (terms not accepted yet?). Run install.bat again later.") }
            }
        } else {
            $Problems.Add("Not logged in to Hugging Face, so MuScriptor and Gemma were skipped. Run install.bat again to add them.")
        }
    }

    Step "Extras: Music Flamingo (its own environment, .venv-listen)"
    try {
        if (-not (Test-Path $PyL)) { Run $Base @("-m", "venv", ".venv-listen") }
        Run $PyL @("-m", "pip", "install", "--upgrade", "pip", "--quiet", "--disable-pip-version-check")
        Run $PyL @("-m", "pip", "install", "torch==2.11.0", "torchaudio==2.11.0", "--index-url", "https://download.pytorch.org/whl/cu128", "--disable-pip-version-check")
        Run $PyL @("-m", "pip", "install", "-r", "installer\requirements-listen.txt", "--disable-pip-version-check")
        $hf = Join-Path $Root "models\hf"
        if (-not $NoModels) {
            Say "  downloading Music Flamingo (about 15 GB)..."
            $env:HF_HOME = $hf
            Run $PyL @("-c", "from huggingface_hub import snapshot_download; snapshot_download('nvidia/music-flamingo-2601-hf')")
            Remove-Item Env:HF_HOME
        }
        # tell Re-Tex where it is (data/config.json, merged over the app's defaults)
        $cfg = @"
import json, pathlib, sys
p = pathlib.Path(r'$Root') / 'data' / 'config.json'
c = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
c['listen'] = {'python': r'$PyL', 'model': 'nvidia/music-flamingo-2601-hf', 'hf_home': r'$hf'}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(c, indent=1), encoding='utf-8')
"@
        Run $Py @("-c", $cfg)
        Say "  Music Flamingo: ready" "Green"
    } catch { $Problems.Add("Music Flamingo didn't finish: $_") }
}

# ------------------------------------------------------------------ shortcut
Step "Shortcut"
if ($NoShortcut) { Say "  skipped" } else { try {
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Re-Tex.lnk"
    $sh = New-Object -ComObject WScript.Shell
    $s = $sh.CreateShortcut($lnk)
    $s.TargetPath = Join-Path $Root "Re-Tex.bat"
    $s.WorkingDirectory = $Root
    $s.IconLocation = Join-Path $Root "web\img\retex.ico"
    $s.Save()
    Say "  Re-Tex shortcut on your desktop" "Green"
} catch { $Problems.Add("Couldn't make the desktop shortcut: $_") } }

# ------------------------------------------------------------------ done
Step "Done"
if ($Problems.Count -eq 0) {
    Say "  Everything is installed. Start Re-Tex with Re-Tex.bat or the desktop shortcut." "Green"
} else {
    Say "  Installed, with these notes:" "Yellow"
    foreach ($p in $Problems) { Say "   - $p" "Yellow" }
    Say "  Re-Tex works without them; run install.bat again any time to finish." "Yellow"
}
