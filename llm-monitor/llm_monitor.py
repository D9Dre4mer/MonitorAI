#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Process Monitor for Prometheus
Detects and monitors LLM processes running on the system
"""

import os
import sys
import time
import subprocess
import psutil
import re
import json
from pathlib import Path
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# Try to import nvidia-ml-py for better GPU process detection
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, OSError):
    NVML_AVAILABLE = False

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Prometheus metrics
llm_process_count = Gauge(
    'llm_process_count',
    'Number of LLM processes detected',
    ['llm_type', 'model_name', 'framework']
)

llm_process_cpu_percent = Gauge(
    'llm_process_cpu_percent',
    'CPU usage percentage per LLM process',
    ['pid', 'name', 'llm_type', 'model_name', 'framework']
)

llm_process_memory_bytes = Gauge(
    'llm_process_memory_bytes',
    'Memory usage in bytes per LLM process',
    ['pid', 'name', 'llm_type', 'model_name', 'framework']
)

llm_process_gpu_memory_bytes = Gauge(
    'llm_process_gpu_memory_bytes',
    'GPU memory usage in bytes per LLM process',
    ['pid', 'name', 'gpu', 'llm_type', 'model_name', 'framework']
)

llm_process_gpu_utilization = Gauge(
    'llm_process_gpu_utilization',
    'GPU utilization percentage per LLM process',
    ['pid', 'name', 'gpu', 'llm_type', 'model_name', 'framework']
)

# LLM detection patterns
LLM_PATTERNS = {
    'transformers': [
        r'transformers',
        r'huggingface',
        r'\.from_pretrained',
        r'pipeline\(.*model',
    ],
    'llama.cpp': [
        r'llama',
        r'llama\.cpp',
        r'gguf',
    ],
    'vllm': [
        r'vllm',
        r'vllm\.engine',
    ],
    'tensorrt': [
        r'tensorrt',
        r'trt',
    ],
    'onnx': [
        r'onnxruntime',
        r'onnx',
    ],
    'pytorch': [
        r'torch',
        r'pytorch',
    ],
    'tensorflow': [
        r'tensorflow',
        r'tf\.',
    ],
}

MODEL_NAME_PATTERNS = [
    r'model[_-]?name["\']?\s*[:=]\s*["\']?([^"\']+)',
    r'--model[_-]?name["\']?\s+([^\s]+)',
    r'--model["\']?\s+([^\s]+)',
    r'model_path["\']?\s*[:=]\s*["\']?([^"\']+)',
    r'/([^/]+)\.(bin|safetensors|gguf|onnx)',
]


def detect_llm_process(process):
    """Detect if a process is running an LLM"""
    try:
        # Check command line
        cmdline = ' '.join(process.cmdline())
        cmdline_lower = cmdline.lower()
        
        # Check if it's a Python process (exclude PowerShell and other non-Python processes)
        process_name_lower = process.name().lower()
        if 'python' not in process_name_lower and 'python' not in cmdline_lower:
            return None
        
        # Exclude PowerShell processes (they might have 'model' in command line but aren't LLM)
        if 'powershell' in process_name_lower or 'powershell' in cmdline_lower:
            return None
        
        # Exclude LLM Monitor and GPU Exporter themselves
        if 'llm_monitor' in cmdline_lower or 'gpu_exporter' in cmdline_lower:
            return None
        
        # Detect LLM type
        llm_type = None
        framework = None
        
        for fw, patterns in LLM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, cmdline_lower, re.IGNORECASE):
                    framework = fw
                    if fw in ['transformers', 'llama.cpp', 'vllm']:
                        llm_type = 'llm'
                    break
            if framework:
                break
        
        # If no specific framework found, check for common LLM indicators
        if not framework:
            llm_indicators = ['model', 'inference', 'generate', 'token', 'embedding']
            if any(indicator in cmdline_lower for indicator in llm_indicators):
                llm_type = 'llm'
                framework = 'unknown'
        
        if not llm_type:
            return None
        
        # Extract model name from multiple sources
        model_name = 'unknown'
        
        # 1. Try to extract from command line arguments (--model-name, --model, etc.)
        for pattern in MODEL_NAME_PATTERNS:
            match = re.search(pattern, cmdline, re.IGNORECASE)
            if match:
                model_name = match.group(1).split('/')[-1].split('\\')[-1]
                # Clean up model name
                model_name = re.sub(r'[^\w\-_.]', '_', model_name)
                if len(model_name) > 50:
                    model_name = model_name[:50]
                break
        
        # 2. Try to extract from environment variables
        if model_name == 'unknown':
            try:
                env = process.environ()
                # Check common environment variable names
                env_vars = [
                    'MODEL_NAME', 'TRANSFORMERS_MODEL_NAME', 'HF_MODEL_NAME',
                    'LLM_MODEL', 'MODEL_ID', 'MODEL_PATH', 'HUGGINGFACE_MODEL'
                ]
                for var in env_vars:
                    if var in env:
                        model_name = env[var].split('/')[-1].split('\\')[-1]
                        model_name = re.sub(r'[^\w\-_.]', '_', model_name)
                        if len(model_name) > 50:
                            model_name = model_name[:50]
                        break
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        
        # 3. Try to extract from file path in command line
        if model_name == 'unknown':
            for part in cmdline.split():
                if any(ext in part.lower() for ext in ['.bin', '.safetensors', '.gguf', '.onnx', '.pt', '.pth']):
                    model_name = os.path.basename(part).split('.')[0]
                    model_name = re.sub(r'[^\w\-_.]', '_', model_name)
                    if len(model_name) > 50:
                        model_name = model_name[:50]
                    break
        
        # 4. Try to extract from script file name or path
        if model_name == 'unknown':
            # Look for model-related file names
            script_patterns = [
                r'([^/\\]+-?model[^/\\]*)\.py',
                r'([^/\\]+-?llm[^/\\]*)\.py',
                r'([^/\\]+-?inference[^/\\]*)\.py',
            ]
            for pattern in script_patterns:
                match = re.search(pattern, cmdline, re.IGNORECASE)
                if match:
                    potential_name = match.group(1)
                    # Skip if it's just "run-llm-model" or similar generic names
                    if potential_name.lower() not in ['run-llm-model', 'llm-model', 'model']:
                        model_name = potential_name.replace('-', '_').replace(' ', '_')
                        model_name = re.sub(r'[^\w\-_.]', '_', model_name)
                        if len(model_name) > 50:
                            model_name = model_name[:50]
                        break
        
        # 5. Try to read from log file if exists (for run-llm-model.py)
        if model_name == 'unknown' and 'run-llm-model' in cmdline_lower:
            log_file = os.path.join('logs', 'llm-model.log')
            if os.path.exists(log_file):
                try:
                    # Read last 50 lines of log file
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-50:]):
                            # Look for "Loading LLM model: ..." or "Model: ..."
                            match = re.search(r'(?:Loading LLM model|Model):\s*([^\s,]+)', line, re.IGNORECASE)
                            if match:
                                model_name = match.group(1).split('/')[-1].split('\\')[-1]
                                model_name = re.sub(r'[^\w\-_.]', '_', model_name)
                                if len(model_name) > 50:
                                    model_name = model_name[:50]
                                break
                except Exception:
                    pass
        
        return {
            'pid': process.pid,
            'name': process.name(),
            'llm_type': llm_type,
            'model_name': model_name,
            'framework': framework,
            'cmdline': cmdline
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None
    except Exception as e:
        print(f"Error detecting LLM process {process.pid}: {e}", file=sys.stderr)
        return None


def get_gpu_processes():
    """Get GPU processes from nvidia-smi, nvidia-ml-py, or GPU info files"""
    gpu_processes = {}
    
    # Method 0: Read GPU info from files exposed by processes (most accurate on Windows)
    try:
        logs_dir = Path('logs')
        if logs_dir.exists():
            for gpu_info_file in logs_dir.glob('gpu-info-*.json'):
                try:
                    with open(gpu_info_file, 'r', encoding='utf-8') as f:
                        gpu_info = json.load(f)
                        pid = gpu_info.get('pid')
                        if pid:
                            # Check if process is still running
                            try:
                                psutil.Process(pid)
                                gpu_index = f"gpu{gpu_info.get('gpu_index', 0)}"
                                memory_bytes = gpu_info.get('gpu_memory_allocated_bytes', 0)
                                memory_mb = memory_bytes / 1024 / 1024
                                utilization = gpu_info.get('gpu_utilization', 0.0)
                                
                                # Always update GPU process info from file (most accurate source)
                                # File is updated by the process itself, so always trust the latest value
                                gpu_processes[pid] = {
                                    'gpu': gpu_index,
                                    'memory_mb': memory_mb,
                                    'utilization': utilization,
                                }
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                # Process no longer exists, remove file
                                try:
                                    gpu_info_file.unlink()
                                except:
                                    pass
                except (json.JSONDecodeError, IOError, KeyError):
                    # Invalid or corrupted file, skip
                    continue
    except Exception as e:
        print(f"Error reading GPU info files: {e}", file=sys.stderr)
    
    # Method 1: Try using nvidia-ml-py (more reliable for detecting GPU memory usage)
    if NVML_AVAILABLE:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_index = f'gpu{i}'
                
                # Get compute processes
                # Note: On Windows, usedGpuMemory is often None, so we track PIDs and get memory from nvidia-smi
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for proc in procs:
                        pid = proc.pid
                        # On Windows, usedGpuMemory is often None, so we use a default or get from nvidia-smi
                        if proc.usedGpuMemory is not None:
                            memory_mb = proc.usedGpuMemory / 1024 / 1024  # Convert bytes to MB
                        else:
                            # Memory will be updated from nvidia-smi query below
                            memory_mb = 0.0
                        
                        if pid not in gpu_processes:
                            gpu_processes[pid] = {
                                'gpu': gpu_index,
                                'memory_mb': memory_mb,
                                'utilization': 0.0,  # Will be updated from nvidia-smi if available
                            }
                        elif proc.usedGpuMemory is not None and memory_mb > gpu_processes[pid]['memory_mb']:
                            gpu_processes[pid]['memory_mb'] = memory_mb
                except pynvml.NVMLError:
                    pass  # No compute processes
                
                # Get graphics processes (may have allocated memory but not actively computing)
                try:
                    procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                    for proc in procs:
                        pid = proc.pid
                        if proc.usedGpuMemory is not None:
                            memory_mb = proc.usedGpuMemory / 1024 / 1024  # Convert bytes to MB
                        else:
                            memory_mb = 0.0
                        
                        if pid not in gpu_processes:
                            gpu_processes[pid] = {
                                'gpu': gpu_index,
                                'memory_mb': memory_mb,
                                'utilization': 0.0,
                            }
                        elif proc.usedGpuMemory is not None and memory_mb > gpu_processes[pid]['memory_mb']:
                            gpu_processes[pid]['memory_mb'] = memory_mb
                except pynvml.NVMLError:
                    pass  # No graphics processes or not supported
        except Exception as e:
            print(f"Error using nvidia-ml-py: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Method 2: Use nvidia-smi to get GPU memory for processes (works better on Windows)
    # This supplements nvidia-ml-py data, especially when usedGpuMemory is None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if not line or '[N/A]' in line or 'Insufficient Permissions' in line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    try:
                        pid = int(parts[0])
                        if parts[1] and parts[1] != '[N/A]':
                            memory_mb = float(parts[1])
                            utilization = float(parts[2]) if parts[2] and parts[2] != '[N/A]' else 0.0
                            gpu_index = 'gpu0'  # Default
                            
                            if pid in gpu_processes:
                                # Update existing entry with memory and utilization from nvidia-smi
                                gpu_processes[pid]['memory_mb'] = memory_mb
                                gpu_processes[pid]['utilization'] = utilization
                            else:
                                # Add new entry
                                gpu_processes[pid] = {
                                    'gpu': gpu_index,
                                    'memory_mb': memory_mb,
                                    'utilization': utilization,
                                }
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Error getting GPU processes from nvidia-smi: {e}", file=sys.stderr)
    
    # Method 3: Fallback if still no processes found
    if not gpu_processes:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory,utilization.gpu,gpu_uuid', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if not line or '[N/A]' in line or 'Insufficient Permissions' in line:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        try:
                            pid = int(parts[0])
                            # Skip if memory is N/A or invalid
                            if parts[2] and parts[2] != '[N/A]':
                                memory_mb = float(parts[2])
                                utilization = float(parts[3]) if parts[3] and parts[3] != '[N/A]' else 0.0
                                gpu_index = 'gpu0'  # Default, can be improved
                                gpu_processes[pid] = {
                                    'gpu': gpu_index,
                                    'memory_mb': memory_mb,
                                    'utilization': utilization,
                                }
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Error getting GPU processes from nvidia-smi: {e}", file=sys.stderr)
    
    return gpu_processes


# Global state to track previous metrics
_previous_pids = set()
_previous_combinations = set()
_previous_pid_info = {}  # pid -> {name, llm_type, model_name, framework, gpu}

def update_metrics():
    """Update Prometheus metrics"""
    global _previous_pids, _previous_combinations, _previous_pid_info
    
    detected_processes = {}
    gpu_processes = get_gpu_processes()
    
    # Debug: Log GPU processes found
    if gpu_processes:
        print(f"DEBUG: Found {len(gpu_processes)} GPU processes: {sorted(gpu_processes.keys())[:10]}", file=sys.stderr)
    
    # Track all detected (llm_type, model_name, framework) combinations
    detected_combinations = set()
    current_pids = set()
    
    # Scan all processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        llm_info = detect_llm_process(proc)
        if llm_info:
            pid = llm_info['pid']
            current_pids.add(pid)
            detected_processes[pid] = llm_info
            # Track this combination
            key = (llm_info['llm_type'], llm_info['model_name'], llm_info['framework'])
            detected_combinations.add(key)
            
            try:
                # Get process metrics
                process = psutil.Process(pid)
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                
                # Store info for this PID
                pid_gpu = None
                if pid in gpu_processes:
                    pid_gpu = gpu_processes[pid]['gpu']
                
                _previous_pid_info[pid] = {
                    'name': llm_info['name'],
                    'llm_type': llm_info['llm_type'],
                    'model_name': llm_info['model_name'],
                    'framework': llm_info['framework'],
                    'gpu': pid_gpu
                }
                
                # Update CPU and memory metrics
                llm_process_cpu_percent.labels(
                    pid=str(pid),
                    name=llm_info['name'],
                    llm_type=llm_info['llm_type'],
                    model_name=llm_info['model_name'],
                    framework=llm_info['framework']
                ).set(cpu_percent)
                
                llm_process_memory_bytes.labels(
                    pid=str(pid),
                    name=llm_info['name'],
                    llm_type=llm_info['llm_type'],
                    model_name=llm_info['model_name'],
                    framework=llm_info['framework']
                ).set(memory_info.rss)
                
                # Update GPU metrics if process is using GPU
                if pid in gpu_processes:
                    gpu_info = gpu_processes[pid]
                    try:
                        # Always set GPU metrics, even if memory is 0 (process is using GPU)
                        memory_bytes = int(gpu_info['memory_mb'] * 1024 * 1024)
                        utilization = gpu_info['utilization']
                        
                        print(f"DEBUG: Updating GPU metrics for PID {pid}: memory={memory_bytes} bytes, utilization={utilization}%", file=sys.stderr)
                        
                        llm_process_gpu_memory_bytes.labels(
                            pid=str(pid),
                            name=llm_info['name'],
                            gpu=gpu_info['gpu'],
                            llm_type=llm_info['llm_type'],
                            model_name=llm_info['model_name'],
                            framework=llm_info['framework']
                        ).set(memory_bytes)
                        
                        llm_process_gpu_utilization.labels(
                            pid=str(pid),
                            name=llm_info['name'],
                            gpu=gpu_info['gpu'],
                            llm_type=llm_info['llm_type'],
                            model_name=llm_info['model_name'],
                            framework=llm_info['framework']
                        ).set(utilization)
                    except Exception as e:
                        print(f"Error updating GPU metrics for PID {pid}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                else:
                    # Debug: Log why PID is not in gpu_processes
                    if pid in [13260, 9484]:  # Only log for known PIDs
                        print(f"DEBUG: PID {pid} not in gpu_processes. GPU processes keys: {sorted(gpu_processes.keys())[:10]}", file=sys.stderr)
                    
                    # Clear GPU metrics if process no longer uses GPU
                    # (process might have stopped using GPU but still running)
                    if pid in _previous_pid_info and _previous_pid_info[pid].get('gpu'):
                        prev_gpu = _previous_pid_info[pid]['gpu']
                        try:
                            llm_process_gpu_memory_bytes.labels(
                                pid=str(pid),
                                name=llm_info['name'],
                                gpu=prev_gpu,
                                llm_type=llm_info['llm_type'],
                                model_name=llm_info['model_name'],
                                framework=llm_info['framework']
                            ).set(0)
                            
                            llm_process_gpu_utilization.labels(
                                pid=str(pid),
                                name=llm_info['name'],
                                gpu=prev_gpu,
                                llm_type=llm_info['llm_type'],
                                model_name=llm_info['model_name'],
                                framework=llm_info['framework']
                            ).set(0)
                        except Exception:
                            pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e:
                print(f"Error updating metrics for PID {pid}: {e}", file=sys.stderr)
    
    # Clear metrics for processes that no longer exist
    dead_pids = _previous_pids - current_pids
    for dead_pid in dead_pids:
        if dead_pid in _previous_pid_info:
            info = _previous_pid_info[dead_pid]
            try:
                # Clear CPU and memory metrics
                llm_process_cpu_percent.labels(
                    pid=str(dead_pid),
                    name=info['name'],
                    llm_type=info['llm_type'],
                    model_name=info['model_name'],
                    framework=info['framework']
                ).set(0)
                
                llm_process_memory_bytes.labels(
                    pid=str(dead_pid),
                    name=info['name'],
                    llm_type=info['llm_type'],
                    model_name=info['model_name'],
                    framework=info['framework']
                ).set(0)
                
                # Clear GPU metrics if process had GPU
                if info['gpu']:
                    llm_process_gpu_memory_bytes.labels(
                        pid=str(dead_pid),
                        name=info['name'],
                        gpu=info['gpu'],
                        llm_type=info['llm_type'],
                        model_name=info['model_name'],
                        framework=info['framework']
                    ).set(0)
                    
                    llm_process_gpu_utilization.labels(
                        pid=str(dead_pid),
                        name=info['name'],
                        gpu=info['gpu'],
                        llm_type=info['llm_type'],
                        model_name=info['model_name'],
                        framework=info['framework']
                    ).set(0)
            except Exception as e:
                # Ignore errors when clearing metrics (label combination might not exist)
                pass
            
            # Remove from tracking
            del _previous_pid_info[dead_pid]
    
    # Update process count by type
    counts = {}
    for proc_info in detected_processes.values():
        key = (proc_info['llm_type'], proc_info['model_name'], proc_info['framework'])
        counts[key] = counts.get(key, 0) + 1
    
    # First, clear all previous combinations that are no longer active
    # This ensures we don't have stale metrics
    dead_combinations = _previous_combinations - detected_combinations
    for (llm_type, model_name, framework) in dead_combinations:
        try:
            llm_process_count.labels(
                llm_type=llm_type,
                model_name=model_name,
                framework=framework
            ).set(0)
        except Exception:
            pass
    
    # Then, set counts for currently detected processes
    # This ensures active processes are always up-to-date
    for (llm_type, model_name, framework), count in counts.items():
        llm_process_count.labels(
            llm_type=llm_type,
            model_name=model_name,
            framework=framework
        ).set(count)
    
    # Note: The above logic already handles the case when all processes stop
    # (dead_combinations will contain all _previous_combinations when detected_combinations is empty)
    # So we don't need a separate check for empty detected_combinations
    
    # Update tracking state
    _previous_pids = current_pids
    _previous_combinations = detected_combinations


def main():
    """Main entry point"""
    port = int(os.getenv('PROMETHEUS_PORT', '9101'))
    
    print(f"Starting LLM Process Monitor on port {port}")
    print("Scanning for LLM processes...")
    
    # Start Prometheus metrics server
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
    
    # Update metrics every 10 seconds
    while True:
        try:
            update_metrics()
        except Exception as e:
            print(f"Error updating metrics: {e}", file=sys.stderr)
        
        time.sleep(10)


if __name__ == '__main__':
    main()

