# Paths for the workspace configuration files
$extensionsPath = ".vscode/extensions.json"
$settingsPath   = ".vscode/settings.json"

# --- STEP 1: Verify Environment ---
if (-not (Test-Path $extensionsPath)) {
    Write-Error "Could not find '$extensionsPath'. Please run this script from the project root."
    exit 1
}

if (-not (Get-Command "code" -ErrorAction SilentlyContinue)) {
    Write-Error "The 'code' command was not found. Please ensure VS Code is installed and added to your Environment PATH."
    exit 1
}

# --- STEP 2: Install Recommended Extensions ---
try {
    $jsonContent = Get-Content -Raw $extensionsPath | ConvertFrom-Json
    $recommendations = $jsonContent.recommendations
} catch {
    Write-Error "Failed to parse '$extensionsPath'. Ensure it is a valid JSON file."
    exit 1
}

if ($null -eq $recommendations -or $recommendations.Count -eq 0) {
    Write-Host "No extensions found under 'recommendations' in $extensionsPath." -ForegroundColor Yellow
} else {
    Write-Host "Found $($recommendations.Count) workspace extension recommendations. Installing..." -ForegroundColor Cyan
    Write-Host "--------------------------------------------------------"
    foreach ($extension in $recommendations) {
        Write-Host "Installing: $extension" -ForegroundColor Green
        code --install-extension $extension --force | Out-Null
    }
    Write-Host "--------------------------------------------------------"
}

# --- STEP 3: Auto-Configure Extension Settings ---
Write-Host "Configuring workspace settings..." -ForegroundColor Cyan

# Enforce your custom Python, Jupyter, and Ruff configurations
$defaultSettings = @{
    "python.defaultInterpreterPath"         = '${workspaceFolder}/.venv/Scripts/python.exe'
    "python.analysis.autoImportCompletions" = $true
    "jupyter.notebookFileRoot"              = '${workspaceFolder}'
    "python.languageServer"                 = "Pylance"
    "python.analysis.typeCheckingMode"      = "basic"
    
    "files.associations"                    = @{
        "*.ipynb" = "jupyter"
    }
    
    "[python]"                              = @{
        "editor.defaultFormatter"  = "charliermarsh.ruff"
        "editor.formatOnSave"      = $true
        "editor.codeActionsOnSave" = @{
            "source.fixAll.ruff"          = "explicit"
            "source.organizeImports.ruff" = "explicit"
        }
    }
}

# Load existing settings or initialize a new object
if (Test-Path $settingsPath) {
    try {
        $currentSettings = Get-Content -Raw $settingsPath | ConvertFrom-Json
        if ($null -eq $currentSettings) { $currentSettings = @{} }
    } catch {
        Write-Warning "Failed to parse '$settingsPath'. Creating a fresh configuration block."
        $currentSettings = @{}
    }
} else {
    # Create the .vscode directory if it doesn't exist
    New-Item -ItemType Directory -Path ".vscode" -Force | Out-Null
    $currentSettings = @{}
}

# Merge default settings without overwriting user-defined overrides
foreach ($key in $defaultSettings.Keys) {
    if (-not $currentSettings.PSObject.Properties[$key]) {
        $currentSettings | Add-Member -NotePropertyName $key -NotePropertyValue $defaultSettings[$key]
        Write-Host "Added setting: $key" -ForegroundColor Gray
    }
}

# Save the updated settings back to .vscode/settings.json
try {
    # Depth 10 ensures nested objects like [python] and files.associations serialize correctly
    $currentSettings | ConvertTo-Json -Depth 10 | Set-Content -Path $settingsPath
    Write-Host "Workspace settings configured successfully!" -ForegroundColor Green
} catch {
    Write-Error "Failed to write updates to '$settingsPath'."
    exit 1
}

Write-Host "--------------------------------------------------------"
Write-Host "All extensions and configurations processed successfully!" -ForegroundColor Cyan