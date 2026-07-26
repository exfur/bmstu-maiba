<#
.SYNOPSIS
    Скрипт автоматической настройки Power BI + Python виртуального окружения (.venv).
.DESCRIPTION
    1. Очищает конфликтующие переменные окружения (PYTHONHOME/PYTHONPATH).
    2. Регистрирует локальный .venv проекта в реестре Windows (PEP 514), указывая на папку \Scripts.
    3. Проверяет функцию контролируемого доступа к папкам Защитника Windows (CFA), чтобы предотвратить блокировку файлов.
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
Write-Host "🚀 Настройка окружения Power BI + Python (.venv)" -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan

# ----------------------------------------------------------------------
# 1. Автоопределение или проверка пути к виртуальному окружению
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
    Write-Host "❌ Ошибка: Не удалось найти python.exe внутри директории .venv!" -ForegroundColor Red
    Write-Host "Пожалуйста, укажите путь к .venv вручную с помощью команды:" -ForegroundColor Yellow
    Write-Host "  .\setup_powerbi.ps1 -VenvPath 'C:\path\to\your\.venv'`n" -ForegroundColor Yellow
    exit 1
}

$scriptsDir = "$VenvPath\Scripts"
$venvPython = "$scriptsDir\python.exe"

Write-Host "✅ Обнаружено виртуальное окружение: $VenvPath" -ForegroundColor Green

# ----------------------------------------------------------------------
# 2. Очистка конфликтующих переменных окружения
# ----------------------------------------------------------------------
Write-Host "`n🧹 Очистка конфликтующих переменных окружения (PYTHONHOME / PYTHONPATH)..." -ForegroundColor Cyan

Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue

[System.Environment]::SetEnvironmentVariable("PYTHONHOME", $null, "User")
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", $null, "User")

Write-Host "✅ Переменные PYTHONHOME и PYTHONPATH очищены для предотвращения конфликтов." -ForegroundColor Green

# ----------------------------------------------------------------------
# 3. Регистрация .venv в реестре Windows для Power BI (PEP 514)
# ----------------------------------------------------------------------
Write-Host "`n📝 Регистрация .venv в реестре Windows для Power BI..." -ForegroundColor Cyan

try {
    $pyVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([string]::IsNullOrWhiteSpace($pyVersion)) {
        throw "Вывод версии Python оказался пустым."
    }
} catch {
    Write-Host "❌ Не удалось выполнить Python из .venv: $_" -ForegroundColor Red
    exit 1
}

$regKey = "HKCU:\Software\Python\PythonCore\$pyVersion-maiba\InstallPath"

New-Item -Path $regKey -Force | Out-Null
Set-ItemProperty -Path $regKey -Name "(default)" -Value "$scriptsDir\"
Set-ItemProperty -Path $regKey -Name "ExecutablePath" -Value $venvPython

Write-Host "✅ В реестре Windows успешно зарегистрирован 'Python $pyVersion-maiba'." -ForegroundColor Green
Write-Host "   Путь: $scriptsDir\" -ForegroundColor Gray

# ----------------------------------------------------------------------
# 4. Проверка контролируемого доступа к папкам Защитника Windows (CFA)
# ----------------------------------------------------------------------
Write-Host "`n🛡️ Проверка Контролируемого доступа к папкам (Защита от вымогателей)..." -ForegroundColor Cyan

try {
    $mpPref = Get-MpPreference -ErrorAction Stop
    $cfaStatus = $mpPref.EnableControlledFolderAccess

    # 1 = Режим блокировки, 2 = Режим аудита
    if ($cfaStatus -eq 1 -or $cfaStatus -eq 2) {
        Write-Host "⚠️ Контролируемый доступ к папкам ВКЛЮЧЕН на этой машине." -ForegroundColor Yellow
        Write-Host "   Эта функция Windows может блокировать доступ Power BI / Python к локальным файлам и моделям." -ForegroundColor Yellow

        $pbiExe = "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"
        $allowedApps = $mpPref.ControlledFolderAccessAllowedApplications

        $pbiAllowed = $allowedApps -contains $pbiExe
        $pyAllowed = $allowedApps -contains $venvPython

        if (-not $pbiAllowed -or -not $pyAllowed) {
            Write-Host "`n💡 Требуется действие (если Power BI выдает ошибки доступа к файлам):" -ForegroundColor White
            Write-Host "   Запустите PowerShell от имени Администратора и выполните:" -ForegroundColor Gray
            if (-not $pbiAllowed) {
                Write-Host "   Add-MpPreference -ControlledFolderAccessAllowedApplications '$pbiExe'" -ForegroundColor Yellow
            }
            if (-not $pyAllowed) {
                Write-Host "   Add-MpPreference -ControlledFolderAccessAllowedApplications '$venvPython'" -ForegroundColor Yellow
            }
        } else {
            Write-Host "✅ Исполняемые файлы Power BI и Python уже находятся в белом списке Defender." -ForegroundColor Green
        }
    } else {
        Write-Host "✅ Контролируемый доступ к папкам отключен (проблем с блокировкой файлов не ожидается)." -ForegroundColor Green
    }
} catch {
    Write-Host "ℹ️ Проверка контролируемого доступа пропущена (используется сторонний антивирус или нестандартная конфигурация Defender)." -ForegroundColor Gray
}

# ----------------------------------------------------------------------
# 5. Инструкция для студентов
# ----------------------------------------------------------------------
Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "🎉 НАСТРОЙКА УСПЕШНО ЗАВЕРШЕНА!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Следующие шаги для студентов:" -ForegroundColor White
Write-Host " 1. Откройте или перезапустите Power BI Desktop." -ForegroundColor Yellow
Write-Host " 2. Перейдите в меню: Файл -> Параметры и настройки -> Параметры -> Скрипты Python." -ForegroundColor Yellow
Write-Host " 3. В поле 'Обнаруженные домашние каталоги Python' выберите 'Python $pyVersion-maiba'." -ForegroundColor Yellow
Write-Host " 4. Нажмите ОК и запускайте ваши Python-скрипты в Power Query!" -ForegroundColor Yellow
Write-Host "========================================================`n"