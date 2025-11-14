# PowerShell script to start all services
# Usage: .\start-all.ps1

param(
    [switch]$ViewLogs,
    [int]$LogTail = 50
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MonitorAI - Start All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check Docker
function Check-Docker {
    Write-Host "[*] Checking Docker..." -ForegroundColor Cyan
    
    try {
        $dockerVersion = docker --version
        Write-Host "[OK] Docker installed: $dockerVersion" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Docker is not installed or not in PATH" -ForegroundColor Red
        return $false
    }
    
    try {
        docker ps | Out-Null
        Write-Host "[OK] Docker daemon is running" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERROR] Docker Desktop is not running!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please start Docker Desktop:" -ForegroundColor Yellow
        Write-Host "  1. Open Docker Desktop application" -ForegroundColor Yellow
        Write-Host "  2. Wait for it to fully start (whale icon in system tray)" -ForegroundColor Yellow
        Write-Host "  3. Run this script again" -ForegroundColor Yellow
        return $false
    }
}

# Function to start Docker services
function Start-DockerServices {
    Write-Host ""
    Write-Host "[*] Starting Docker services..." -ForegroundColor Cyan
    docker-compose up -d
    
    Write-Host ""
    Write-Host "[*] Waiting for services to be ready (30 seconds)..." -ForegroundColor Cyan
    Start-Sleep -Seconds 30
    
    Write-Host ""
    Write-Host "[*] Service status:" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host ""
    Write-Host "[OK] Docker services started!" -ForegroundColor Green
}

# Function to start LLM Monitor
function Start-LLMMonitor {
    Write-Host ""
    Write-Host "[*] Starting LLM Monitor..." -ForegroundColor Cyan
    
    # Check if already running
    $existing = Get-Process python -ErrorAction SilentlyContinue | Where-Object { 
        $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
        $cmdline -like "*llm_monitor*"
    }
    if ($existing) {
        Write-Host "[WARN] LLM Monitor already running (PID: $($existing.Id)). Skipping..." -ForegroundColor Yellow
        return
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Python not found!" -ForegroundColor Red
        return
    }
    
    # Install dependencies
    Write-Host "[*] Installing dependencies..." -ForegroundColor Cyan
    Push-Location llm-monitor
    pip install -q -r requirements.txt 2>&1 | Out-Null
    Pop-Location
    
    # Set environment and start
    $env:PROMETHEUS_PORT = "9101"
    Write-Host "[*] Starting LLM Process Monitor on port 9101..." -ForegroundColor Cyan
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:PROMETHEUS_PORT='9101'; python llm-monitor\llm_monitor.py" -WindowStyle Minimized
    
    Write-Host "[OK] LLM Monitor started in background" -ForegroundColor Green
}

# Function to start GPU Exporter
function Start-GPUExporter {
    Write-Host ""
    Write-Host "[*] Starting GPU Exporter..." -ForegroundColor Cyan
    
    # Check if already running
    $existing = Get-Process python -ErrorAction SilentlyContinue | Where-Object { 
        $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
        $cmdline -like "*gpu_exporter*"
    }
    if ($existing) {
        Write-Host "[WARN] GPU Exporter already running (PID: $($existing.Id)). Skipping..." -ForegroundColor Yellow
        return
    }
    
    # Check nvidia-smi
    try {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Found GPU: $gpuInfo" -ForegroundColor Green
        } else {
            Write-Host "[WARN] nvidia-smi not available - GPU Exporter may not work" -ForegroundColor Yellow
            return
        }
    } catch {
        Write-Host "[WARN] nvidia-smi not found - GPU Exporter may not work" -ForegroundColor Yellow
        return
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Python not found!" -ForegroundColor Red
        return
    }
    
    # Install dependencies
    Write-Host "[*] Installing dependencies..." -ForegroundColor Cyan
    Push-Location gpu-exporter
    pip install -q prometheus-client 2>&1 | Out-Null
    Pop-Location
    
    # Set environment and start
    $env:SERVICE_NAME = "llm-monitor"
    $env:NAMESPACE = "default"
    $env:PROMETHEUS_PORT = "9100"
    Write-Host "[*] Starting GPU metrics exporter on port 9100..." -ForegroundColor Cyan
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:SERVICE_NAME='llm-monitor'; `$env:NAMESPACE='default'; `$env:PROMETHEUS_PORT='9100'; python gpu-exporter\gpu_exporter.py" -WindowStyle Minimized
    
    Write-Host "[OK] GPU Exporter started in background" -ForegroundColor Green
}

# Function to view logs
function View-Logs {
    $logFile = "logs/llm-model.log"
    
    if (-not (Test-Path $logFile)) {
        Write-Host "[WARN] Log file not found: $logFile" -ForegroundColor Yellow
        Write-Host "  Run the LLM model first to generate logs" -ForegroundColor Gray
        return
    }
    
    Write-Host ""
    Write-Host "[*] Viewing LLM Model Logs" -ForegroundColor Cyan
    Write-Host "  File: $logFile" -ForegroundColor Gray
    Write-Host ""
    Write-Host "[*] Last $LogTail lines:" -ForegroundColor Yellow
    Get-Content $logFile -Tail $LogTail
}

# Main execution
if ($ViewLogs) {
    View-Logs
    exit 0
}

# Start all services
if (-not (Check-Docker)) {
    exit 1
}

Start-DockerServices
Start-LLMMonitor
Start-GPUExporter

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "[*] Access Grafana at: http://localhost:3000" -ForegroundColor Cyan
Write-Host "   Username: admin" -ForegroundColor Gray
Write-Host "   Password: admin" -ForegroundColor Gray
Write-Host ""
Write-Host "[*] Other services:" -ForegroundColor Cyan
Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor Gray
Write-Host "   Loki: http://localhost:3100" -ForegroundColor Gray
Write-Host "   Tempo: http://localhost:3200" -ForegroundColor Gray
Write-Host ""
Write-Host "[*] Monitoring services:" -ForegroundColor Cyan
Write-Host "   LLM Monitor: http://localhost:9101/metrics" -ForegroundColor Gray
Write-Host "   GPU Exporter: http://localhost:9100/metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "[*] Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run LLM model: python run-llm-model.py --model-name microsoft/DialoGPT-small" -ForegroundColor Yellow
Write-Host "  2. View logs: .\start-all.ps1 -ViewLogs" -ForegroundColor Yellow
Write-Host "  3. Open Grafana dashboard: http://localhost:3000" -ForegroundColor Yellow
Write-Host ""

