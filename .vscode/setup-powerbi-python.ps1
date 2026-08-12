<#
.SYNOPSIS
    Automated setup script for Power BI + Python virtual environment (.venv).
.DESCRIPTION
    1. Clears conflicting environment variables (PYTHONHOME/PYTHONPATH).
    2. Registers the project's local .venv in the Windows Registry (PEP 514), pointing to the \Scripts directory.
    3. Checks Windows Defender Controlled Folder Access (CFA) to prevent file blocking.
.EXAMPLE
    .\setup_powerbi.ps1
.EXAMPLE
    .\setup_powerbi.ps1 -VenvPath "C:\custom\path\.venv"
#>

[CmdletBinding()]
param (
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "🚀 Power BI + Python (.venv) Environment Setup" -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan

# ----------------------------------------------------------------------
# 1. Auto-detection or verification of virtual environment path
# ----------------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $candidatePaths = @(
        "$PSScriptRoot\.venv",
        "$PSScriptRoot\dev\.venv",
        "$PSScriptRoot\..\.venv"
    )
    foreach ($path in $candidatePaths) {
        if (Test-Path "$path\Scripts\python.exe") {
            $VenvPath = (Get-Item "$path").FullName
            break
        }
    }
}

if (-not $VenvPath -or -not (Test-Path "$VenvPath\Scripts\python.exe")) {
    Write-Host "❌ Error: Could not find python.exe inside the .venv directory!" -ForegroundColor Red
    Write-Host "Please specify the path to .venv manually using the command:" -ForegroundColor Yellow
    Write-Host "  .\setup_powerbi.ps1 -VenvPath 'C:\path\to\your\.venv'`n" -ForegroundColor Yellow
    exit 1
}

$scriptsDir = "$VenvPath\Scripts"
$venvPython = "$scriptsDir\python.exe"

Write-Host "✅ Virtual environment detected: $VenvPath" -ForegroundColor Green

# ----------------------------------------------------------------------
# 2. Clean up conflicting environment variables
# ----------------------------------------------------------------------
Write-Host "`n🧹 Cleaning up conflicting environment variables (PYTHONHOME / PYTHONPATH)..." -ForegroundColor Cyan

Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue

[System.Environment]::SetEnvironmentVariable("PYTHONHOME", $null, "User")
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", $null, "User")

Write-Host "✅ PYTHONHOME and PYTHONPATH environment variables cleared to prevent conflicts." -ForegroundColor Green

# ----------------------------------------------------------------------
# 3. Register .venv in Windows Registry for Power BI (PEP 514)
# ----------------------------------------------------------------------
Write-Host "`n📝 Registering .venv in Windows Registry for Power BI..." -ForegroundColor Cyan

try {
    $pyVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([string]::IsNullOrWhiteSpace($pyVersion)) {
        throw "Python version output was empty."
    }
} catch {
    Write-Host "❌ Failed to execute Python from .venv: $_" -ForegroundColor Red
    exit 1
}

$regKey = "HKCU:\Software\Python\PythonCore\${pyVersion}-maiba\InstallPath"

New-Item -Path $regKey -Force | Out-Null
Set-ItemProperty -Path $regKey -Name "(default)" -Value "$scriptsDir\"
Set-ItemProperty -Path $regKey -Name "ExecutablePath" -Value $venvPython

Write-Host "✅ Successfully registered 'Python ${pyVersion}-maiba' in Windows Registry." -ForegroundColor Green
Write-Host "   Path: $scriptsDir\" -ForegroundColor Gray

# ----------------------------------------------------------------------
# 4. Check Windows Defender Controlled Folder Access (CFA)
# ----------------------------------------------------------------------
Write-Host "`n🛡️ Checking Controlled Folder Access (Ransomware Protection)..." -ForegroundColor Cyan

try {
    $mpPref = Get-MpPreference -ErrorAction Stop
    $cfaStatus = $mpPref.EnableControlledFolderAccess

    # 1 = Block Mode, 2 = Audit Mode
    if ($cfaStatus -eq 1 -or $cfaStatus -eq 2) {
        Write-Host "⚠️ Controlled Folder Access is ENABLED on this machine." -ForegroundColor Yellow
        Write-Host "   This Windows feature may block Power BI / Python access to local files and models." -ForegroundColor Yellow

        $pbiExe = "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"
        $allowedApps = $mpPref.ControlledFolderAccessAllowedApplications

        $pbiAllowed = $allowedApps -contains $pbiExe
        $pyAllowed = $allowedApps -contains $venvPython

        if (-not $pbiAllowed -or -not $pyAllowed) {
            Write-Host "`n💡 Action required (if Power BI throws file access errors):" -ForegroundColor White
            Write-Host "   Run PowerShell as Administrator and execute:" -ForegroundColor Gray
            if (-not $pbiAllowed) {
                Write-Host "   Add-MpPreference -ControlledFolderAccessAllowedApplications '$pbiExe'" -ForegroundColor Yellow
            }
            if (-not $pyAllowed) {
                Write-Host "   Add-MpPreference -ControlledFolderAccessAllowedApplications '$venvPython'" -ForegroundColor Yellow
            }
        } else {
            Write-Host "✅ Power BI and Python executables are already whitelisted in Defender." -ForegroundColor Green
        }
    } else {
        Write-Host "✅ Controlled Folder Access is disabled (no file blocking issues expected)." -ForegroundColor Green
    }
} catch {
    Write-Host "ℹ️ Controlled Folder Access check skipped (third-party antivirus in use or non-standard Defender configuration)." -ForegroundColor Gray
}

# ----------------------------------------------------------------------
# 5. Instructions for students
# ----------------------------------------------------------------------
Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "🎉 SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Next steps for students:" -ForegroundColor White
Write-Host " 1. Open or restart Power BI Desktop." -ForegroundColor Yellow
Write-Host " 2. Navigate to: File -> Options and settings -> Options -> Python scripting." -ForegroundColor Yellow
Write-Host " 3. Under 'Detected Python home directories', select 'Python ${pyVersion}-maiba'." -ForegroundColor Yellow
Write-Host " 4. Click OK and run your Python scripts in Power Query!" -ForegroundColor Yellow
Write-Host "========================================================`n"