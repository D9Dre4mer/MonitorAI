# PowerShell script to stop all services
# Usage: .\stop-all.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MonitorAI - Stop All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Stop Docker services
Write-Host "[*] Stopping Docker services..." -ForegroundColor Cyan
docker-compose down
Write-Host "[OK] Docker services stopped" -ForegroundColor Green

# Stop LLM Monitor
Write-Host ""
Write-Host "[*] Stopping LLM Monitor..." -ForegroundColor Cyan
$llmMonitor = Get-Process python -ErrorAction SilentlyContinue | Where-Object { 
    $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    $cmdline -like "*llm_monitor*"
}
if ($llmMonitor) {
    foreach ($proc in $llmMonitor) {
        Write-Host "  Stopping PID $($proc.Id)..." -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force
    }
    Write-Host "[OK] LLM Monitor stopped" -ForegroundColor Green
} else {
    Write-Host "[INFO] LLM Monitor not running" -ForegroundColor Gray
}

# Stop GPU Exporter
Write-Host ""
Write-Host "[*] Stopping GPU Exporter..." -ForegroundColor Cyan
$gpuExporter = Get-Process python -ErrorAction SilentlyContinue | Where-Object { 
    $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    $cmdline -like "*gpu_exporter*"
}
if ($gpuExporter) {
    foreach ($proc in $gpuExporter) {
        Write-Host "  Stopping PID $($proc.Id)..." -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force
    }
    Write-Host "[OK] GPU Exporter stopped" -ForegroundColor Green
} else {
    Write-Host "[INFO] GPU Exporter not running" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All services stopped!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

