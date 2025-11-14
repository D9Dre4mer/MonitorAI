# MonitorAI - LLM Monitoring Stack - Green Forest Theme

Monitoring stack để theo dõi các LLM (Large Language Model) processes đang chạy trên máy với Grafana, Prometheus và dashboard Green Forest theme.

## 📋 Mô tả

Hệ thống tự động phát hiện và theo dõi các LLM processes (transformers, llama.cpp, vLLM, TensorRT, ONNX, PyTorch, TensorFlow) với metrics:
- CPU và Memory usage per process
- GPU utilization và VRAM usage per process (hỗ trợ Windows qua file JSON)
- Process count theo framework và model name
- Logs aggregation qua Loki
- Tự động clear metrics cho processes đã dừng

## 🔄 Pipeline Hệ Thống

Hệ thống monitoring hoạt động theo 3 pipeline chính:

### 1. Metrics Pipeline (CPU, Memory, GPU)

```
┌─────────────────┐
│  LLM Processes │
│  (Python apps) │
└────────┬────────┘
         │
         ├─► CPU/Memory metrics ──┐
         │                        │
         └─► GPU info (JSON) ─────┤
            logs/gpu-info-*.json  │
                                  │
         ┌────────────────────────▼────────┐
         │      LLM Monitor (9101)         │
         │  - Detect LLM processes         │
         │  - Read GPU info files          │
         │  - Collect CPU/Memory/GPU       │
         │  - Expose Prometheus metrics    │
         └──────────────┬──────────────────┘
                        │
         ┌──────────────▼──────────────────┐
         │   GPU Exporter (9100)          │
         │  - Query nvidia-smi             │
         │  - Collect overall GPU metrics  │
         │  - Expose Prometheus metrics    │
         └──────────────┬──────────────────┘
                        │
         ┌──────────────▼──────────────────┐
         │      Prometheus (9090)          │
         │  - Scrape metrics every 15s     │
         │  - Store time-series data       │
         │  - Retention: 200 hours         │
         └──────────────┬──────────────────┘
                        │
         ┌──────────────▼──────────────────┐
         │      Grafana (3000)             │
         │  - Query Prometheus via PromQL  │
         │  - Visualize in dashboards      │
         │  - Green Forest theme           │
         └──────────────────────────────────┘
```

**Chi tiết:**
- **LLM Processes** chạy và ghi GPU memory vào `logs/gpu-info-{PID}.json` (mỗi 10 giây)
- **LLM Monitor** (port 9101):
  - Quét tất cả processes mỗi 10 giây
  - Đọc GPU info files từ `logs/` directory
  - Thu thập CPU, Memory, GPU metrics per process
  - Expose metrics qua Prometheus client library
- **GPU Exporter** (port 9100):
  - Query `nvidia-smi` mỗi 15 giây
  - Thu thập overall GPU metrics (utilization, memory, temperature, power)
  - Expose metrics qua Prometheus client library
- **Prometheus** (port 9090):
  - Scrape LLM Monitor và GPU Exporter mỗi 15 giây
  - Lưu trữ time-series data với retention 200 giờ
  - Cung cấp PromQL để query metrics
- **Grafana** (port 3000):
  - Kết nối đến Prometheus qua datasource
  - Hiển thị metrics trong dashboard với Green Forest theme
  - Auto-refresh mỗi 15 giây

### 2. Logs Pipeline

```
┌─────────────────┐
│  LLM Processes │
│  (Python apps)  │
└────────┬────────┘
         │
         │ Write logs
         ▼
┌─────────────────┐
│ logs/llm-model. │
│      log        │
└────────┬────────┘
         │
         │ Read logs
         ▼
┌─────────────────┐
│   Promtail      │
│  (Log Shipper)  │
│  - Tail log file│
│  - Parse & label│
└────────┬────────┘
         │
         │ Push logs
         ▼
┌─────────────────┐
│   Loki (3100)   │
│  - Store logs   │
│  - Index by     │
│    labels       │
└────────┬────────┘
         │
         │ Query logs
         ▼
┌─────────────────┐
│   Grafana       │
│  - Logs panel   │
│  - LogQL queries│
└─────────────────┘
```

**Chi tiết:**
- **LLM Processes** ghi logs vào `logs/llm-model.log` (format: timestamp, level, message)
- **Promtail** (Docker container):
  - Đọc log file từ `logs/` directory (mounted volume)
  - Parse và label logs với `job=llm-model`
  - Push logs đến Loki qua HTTP API
- **Loki** (port 3100):
  - Nhận và lưu trữ logs
  - Index logs theo labels để query nhanh
  - Cung cấp LogQL để query logs
- **Grafana**:
  - Kết nối đến Loki qua datasource
  - Hiển thị logs trong Logs panel
  - Hỗ trợ LogQL queries và filtering

### 3. Tracing Pipeline (Optional - Future)

```
┌─────────────────┐
│  Applications   │
│  (OpenTelemetry)│
└────────┬────────┘
         │
         │ Send traces
         ▼
┌─────────────────┐
│  Tempo (3200)    │
│  - Store traces  │
│  - OTLP protocol │
└────────┬────────┘
         │
         │ Query traces
         ▼
┌─────────────────┐
│   Grafana       │
│  - Trace view   │
│  - Flame graphs │
└─────────────────┘
```

**Chi tiết:**
- **Tempo** (port 3200) sẵn sàng nhận traces qua OTLP (gRPC port 4317, HTTP port 4318)
- Hiện tại chưa có application gửi traces, nhưng infrastructure đã sẵn sàng
- Có thể tích hợp OpenTelemetry SDK vào LLM processes để gửi traces

### Data Flow Summary

| Component | Input | Output | Frequency |
|-----------|-------|--------|-----------|
| LLM Processes | - | GPU info JSON, Logs | 10s (inference) |
| LLM Monitor | Processes, GPU JSON | Prometheus metrics | 10s |
| GPU Exporter | nvidia-smi | Prometheus metrics | 15s |
| Promtail | Log files | Loki logs | Real-time |
| Prometheus | Metrics endpoints | Time-series DB | 15s scrape |
| Loki | Promtail logs | Log storage | Real-time |
| Grafana | Prometheus, Loki | Dashboard | 15s refresh |

## 🚀 Hướng dẫn chạy

### Yêu cầu
- Docker Desktop đang chạy
- Conda environment `Grafotel` với Python 3.11+
- NVIDIA GPU với nvidia-smi (optional, cho GPU metrics)
- PyTorch với CUDA support (cho GPU monitoring chính xác)

### Bước 1: Start Tất Cả Services

```powershell
.\start-all.ps1
```

Script này sẽ tự động khởi động:
- Docker services: Grafana (3000), Prometheus (9090), Loki (3100), Tempo (3200)
- LLM Monitor (port 9101) - chạy background
- GPU Exporter (port 9100) - chạy background (nếu có NVIDIA GPU)

### Bước 2: Chạy LLM Model

**Option 1: Chạy model với GPU (khuyến nghị)**

Mở terminal mới, kích hoạt conda environment và chạy:

```powershell
conda activate Grafotel
python run-llm-model-gpu.py
```

Script này sẽ:
- Tự động detect GPU và load model lên GPU
- Expose GPU memory usage qua file JSON (`logs/gpu-info-{PID}.json`)
- LLM Monitor sẽ đọc file này để lấy GPU metrics chính xác

**Option 2: Chạy model CPU hoặc GPU (tự động detect)**

```powershell
conda activate Grafotel
python run-llm-model.py --model-name microsoft/DialoGPT-small
```

LLM Monitor sẽ tự động detect model và collect metrics (CPU, Memory, GPU nếu có).

### Bước 3: Xem Dashboard

- Truy cập: http://localhost:3000
- Login: `admin` / `admin`
- Dashboard: **Dashboards** → **LLM Monitoring – Green Forest Dashboard**

## 🛑 Dừng services

```powershell
.\stop-all.ps1
```

Dừng tất cả: Docker services, LLM Monitor, GPU Exporter

## 📋 Xem Logs

```powershell
.\start-all.ps1 -ViewLogs
```

Logs được lưu trong thư mục `logs/`:
- `logs/llm-model.log` - Logs từ LLM models
- `logs/gpu-info-{PID}.json` - GPU memory info từ processes (tự động tạo)

## 🧹 Reset Docker (xóa tất cả data)

```powershell
.\stop-all.ps1
docker-compose down -v
```

## 📁 Cấu trúc project

```
MonitorAI/
├── config/                  # Config files cho Grafana, Prometheus, Loki, Tempo, Promtail
│   ├── grafana/            # Grafana datasources và dashboard provider
│   ├── prometheus/         # Prometheus config
│   ├── loki/               # Loki config
│   ├── tempo/              # Tempo config
│   └── promtail/           # Promtail config (log shipping)
├── dashboards/              # LLM monitoring dashboard
│   └── llm-monitoring-green-forest.json
├── llm-monitor/             # LLM Process Monitor
│   ├── llm_monitor.py      # Main monitor script
│   └── requirements.txt    # Python dependencies (bao gồm nvidia-ml-py)
├── gpu-exporter/            # GPU Metrics Exporter
│   ├── gpu_exporter.py     # GPU metrics collector
│   └── requirements.txt
├── logs/                    # Logs directory (tự động tạo)
│   ├── llm-model.log       # LLM model logs
│   └── gpu-info-*.json     # GPU info files (tự động tạo bởi processes)
├── docker-compose.yml       # Docker services
├── start-all.ps1            # Start all services (Docker + LLM Monitor + GPU Exporter)
├── stop-all.ps1             # Stop all services
├── run-llm-model.py         # Example LLM model script (CPU/GPU auto-detect)
└── run-llm-model-gpu.py    # GPU-optimized LLM model script
```

## 🔍 LLM Detection

Tự động phát hiện các frameworks:
- Hugging Face Transformers
- llama.cpp
- vLLM
- TensorRT
- ONNX Runtime
- PyTorch
- TensorFlow

Model names được tự động extract từ command line hoặc file paths.

## 🎯 GPU Monitoring (Windows)

Trên Windows, GPU monitoring sử dụng cơ chế **file-based exposure** để đảm bảo độ chính xác:

1. **Process tự expose GPU memory:**
   - `run-llm-model-gpu.py` và `run-llm-model.py` ghi GPU memory vào `logs/gpu-info-{PID}.json`
   - File được cập nhật mỗi lần inference (mỗi 10 giây)
   - Chứa: `pid`, `gpu_memory_allocated_bytes`, `gpu_memory_reserved_bytes`, `gpu_utilization`, `gpu_index`, `timestamp`

2. **LLM Monitor đọc file JSON:**
   - Đọc tất cả file `logs/gpu-info-*.json` mỗi 10 giây
   - Lấy GPU memory chính xác từ PyTorch (thay vì dựa vào `nvidia-smi` có thể không chính xác trên Windows)
   - Tự động xóa file nếu process không còn tồn tại

3. **Metrics được expose:**
   - `llm_process_gpu_memory_bytes` - GPU memory usage per process
   - `llm_process_gpu_utilization` - GPU utilization per process
   - `nvidia_gpu_utilization` - Overall GPU utilization
   - `nvidia_gpu_memory_used_bytes` - Overall GPU memory used
   - `nvidia_gpu_memory_total_bytes` - Total GPU memory
   - `nvidia_gpu_temperature` - GPU temperature
   - `nvidia_gpu_power_usage` - GPU power usage

## 🐛 Troubleshooting

**LLM Monitor không detect processes:**
- Kiểm tra có Python processes đang chạy: `Get-Process python`
- Verify LLM Monitor đang chạy: `curl http://localhost:9101/metrics`
- Đảm bảo đang chạy trong conda environment `Grafotel`

**Dashboard không có data:**
- Đảm bảo LLM Monitor đang chạy
- Kiểm tra time range (Last 15 minutes)
- Verify Prometheus có metrics: http://localhost:9090/graph?g0.expr=llm_process_count
- Kiểm tra Prometheus targets: http://localhost:9090/targets

**GPU metrics không hiện hoặc hiện 0.0:**
- Kiểm tra GPU Exporter: `curl http://localhost:9100/metrics`
- Verify nvidia-smi hoạt động: `nvidia-smi`
- Kiểm tra PyTorch có CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Kiểm tra file GPU info: `Get-ChildItem logs\gpu-info-*.json`
- Đảm bảo model đang chạy với GPU: sử dụng `run-llm-model-gpu.py`
- Restart LLM Monitor sau khi model bắt đầu chạy

**Process đã dừng nhưng vẫn hiện trên dashboard:**
- LLM Monitor tự động clear metrics sau 10 giây
- Nếu vẫn hiện, restart LLM Monitor: `.\stop-all.ps1` rồi `.\start-all.ps1`

**Logs không hiện trong Grafana:**
- Kiểm tra Promtail đang chạy: `docker ps | Select-String promtail`
- Verify logs file tồn tại: `Test-Path logs\llm-model.log`
- Kiểm tra Loki: http://localhost:3100/ready

## 📝 License

MIT License
