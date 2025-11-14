# PowerShell script to verify MonitorAI setup
# Usage: .\verify-setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MonitorAI - Setup Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Check Docker
Write-Host "[*] Checking Docker..." -ForegroundColor Cyan
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker not found!" -ForegroundColor Red
    $errors++
}

# Check Docker Compose
Write-Host "[*] Checking Docker Compose..." -ForegroundColor Cyan
try {
    $composeVersion = docker-compose --version
    Write-Host "[OK] Docker Compose: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker Compose not found!" -ForegroundColor Red
    $errors++
}

# Check Python
Write-Host "[*] Checking Python..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version
    Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    $errors++
}

# Check Conda
Write-Host "[*] Checking Conda..." -ForegroundColor Cyan
try {
    $condaVersion = conda --version
    Write-Host "[OK] Conda: $condaVersion" -ForegroundColor Green
    
    # Check Grafotel environment
    $envExists = conda env list | Select-String "Grafotel"
    if ($envExists) {
        Write-Host "[OK] Conda environment 'Grafotel' exists" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Conda environment 'Grafotel' not found" -ForegroundColor Yellow
        $warnings++
    }
} catch {
    Write-Host "[WARN] Conda not found (optional)" -ForegroundColor Yellow
    $warnings++
}

# Check NVIDIA GPU (optional)
Write-Host "[*] Checking NVIDIA GPU..." -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] GPU found: $gpuInfo" -ForegroundColor Green
    } else {
        Write-Host "[WARN] nvidia-smi not available (GPU monitoring will be limited)" -ForegroundColor Yellow
        $warnings++
    }
} catch {
    Write-Host "[WARN] nvidia-smi not found (GPU monitoring will be limited)" -ForegroundColor Yellow
    $warnings++
}

# Check required files
Write-Host "[*] Checking required files..." -ForegroundColor Cyan
$requiredFiles = @(
    "docker-compose.yml",
    "README.md",
    "start-all.ps1",
    "stop-all.ps1",
    "llm-monitor/llm_monitor.py",
    "gpu-exporter/gpu_exporter.py",
    "config/prometheus/prometheus.yml",
    "config/loki/loki-config.yml",
    "config/tempo/tempo-config.yml",
    "config/promtail/promtail-config.yml",
    "dashboards/llm-monitoring-green-forest.json"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "[OK] Found: $file" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Missing: $file" -ForegroundColor Red
        $errors++
    }
}

# Check logs directory
Write-Host "[*] Checking logs directory..." -ForegroundColor Cyan
if (Test-Path "logs") {
    Write-Host "[OK] Logs directory exists" -ForegroundColor Green
} else {
    Write-Host "[INFO] Logs directory will be created automatically" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "  Setup verification: PASSED" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now start MonitorAI with:" -ForegroundColor Cyan
    Write-Host "  .\start-all.ps1" -ForegroundColor Yellow
    exit 0
} elseif ($errors -eq 0) {
    Write-Host "  Setup verification: PASSED (with warnings)" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Warnings: $warnings" -ForegroundColor Yellow
    Write-Host "You can proceed, but some features may be limited." -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "  Setup verification: FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Errors: $errors" -ForegroundColor Red
    Write-Host "Warnings: $warnings" -ForegroundColor Yellow
    Write-Host "Please fix the errors before proceeding." -ForegroundColor Red
    exit 1
}

