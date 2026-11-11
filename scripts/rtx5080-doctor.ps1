# Read-only environment probe for the K3 RTX 5080 fork.
# This script does not install software, change drivers, modify firmware, or alter GPU settings.

$ErrorActionPreference = "Stop"

function Section([string]$Name) {
    Write-Host ""
    Write-Host "=== $Name ==="
}

function Have([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Section "CPU / RAM / OS"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs  = Get-CimInstance Win32_ComputerSystem
$os  = Get-CimInstance Win32_OperatingSystem
Write-Host ("CPU: {0} | Cores: {1} | Logical: {2}" -f $cpu.Name.Trim(), $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors)
Write-Host ("RAM: {0:N1} GiB" -f ($cs.TotalPhysicalMemory / 1GB))
Write-Host ("OS : {0} {1} build {2}" -f $os.Caption, $os.OSArchitecture, $os.BuildNumber)

Section "NVIDIA"
if (Have "nvidia-smi") {
    & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,pci.bus_id,pstate,power.limit --format=csv,noheader
} else {
    Write-Warning "nvidia-smi not found on PATH."
}

Section "CUDA compiler"
if (Have "nvcc") {
    & nvcc --version
} else {
    Write-Warning "nvcc not found. The upstream CPU build can still work, but the CUDA probe cannot be built yet."
}

Section "CMake"
if (Have "cmake") {
    (& cmake --version | Select-Object -First 1)
} else {
    Write-Warning "cmake not found."
}

Section "MSVC / Visual Studio"
if (Have "cl") {
    Write-Host "cl.exe is available in this shell."
    & cl 2>&1 | Select-Object -First 2
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) {
            Write-Host "Visual Studio C++ tools found at: $vs"
            Write-Host "cl.exe is not active in this shell; use a Developer PowerShell/Command Prompt when building CUDA."
        } else {
            Write-Warning "Visual Studio found, but the C++ toolchain requirement was not detected."
        }
    } else {
        Write-Warning "MSVC/Visual Studio C++ tools were not detected."
    }
}

Section "Fixed storage"
$drives = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
    Select-Object DeviceID, Size, FreeSpace

foreach ($d in $drives) {
    Write-Host ("{0}  size={1:N2} TiB  free={2:N2} TiB" -f
        $d.DeviceID, ($d.Size / 1TB), ($d.FreeSpace / 1TB))
}

# Published checkpoint bytes plus conservative filesystem/headroom margins.
$checkpointNeed = 1.61TB
$trunkNeed = 0.13TB
$singleDriveNeed = 1.78TB

$single = $drives | Where-Object { $_.FreeSpace -ge $singleDriveNeed }
$checkpointDrives = $drives | Where-Object { $_.FreeSpace -ge $checkpointNeed }
$trunkDrives = $drives | Where-Object { $_.FreeSpace -ge $trunkNeed }

Section "Full K3 storage gate"
if ($single) {
    Write-Host "PASS: at least one fixed volume has enough free space for checkpoint + packed trunk + margin."
} elseif ($checkpointDrives -and $trunkDrives.Count -ge 2) {
    Write-Host "PASS (split layout): one volume can hold the checkpoint and another can hold the packed trunk."
} else {
    Write-Warning "FAIL: current fixed-volume free space is not sufficient for the full K3 deployment."
    Write-Host "Do not start the 1.56 TB checkpoint download yet."
}

Section "Safety"
Write-Host "No settings were changed. No driver, VBIOS, firmware, clocks, voltages, or power limits were modified."
