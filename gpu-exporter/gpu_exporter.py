#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Metrics Exporter for Prometheus
Reads GPU metrics from nvidia-smi and exports to Prometheus
"""

import os
import sys
import time
import subprocess
import re
from prometheus_client import Gauge, start_http_server

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Prometheus metrics
nvidia_gpu_utilization = Gauge(
    'nvidia_gpu_utilization',
    'GPU utilization percentage',
    ['gpu', 'gpu_type', 'service', 'namespace']
)

nvidia_gpu_memory_used_bytes = Gauge(
    'nvidia_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu', 'gpu_type', 'service', 'namespace']
)

nvidia_gpu_memory_total_bytes = Gauge(
    'nvidia_gpu_memory_total_bytes',
    'GPU total memory in bytes',
    ['gpu', 'gpu_type', 'service', 'namespace']
)

nvidia_gpu_temperature = Gauge(
    'nvidia_gpu_temperature',
    'GPU temperature in Celsius',
    ['gpu', 'gpu_type', 'service', 'namespace']
)

nvidia_gpu_power_usage = Gauge(
    'nvidia_gpu_power_usage',
    'GPU power usage in watts',
    ['gpu', 'gpu_type', 'service', 'namespace']
)


def get_gpu_info():
    """Get GPU information from nvidia-smi"""
    try:
        # Get GPU name and count
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                # Split by comma and strip whitespace
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    gpus.append({
                        'index': parts[0],
                        'name': parts[1]
                    })
        
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}", file=sys.stderr)
        return []


def get_gpu_metrics(gpu_index, gpu_name):
    """Get metrics for a specific GPU"""
    try:
        # Query all GPUs and filter by index
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        # Parse output: index, name, utilization.gpu, memory.used, memory.total, temperature.gpu, power.draw
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7 and parts[0] == str(gpu_index):
                return {
                    'utilization': float(parts[2]),
                    'memory_used_mb': float(parts[3]),
                    'memory_total_mb': float(parts[4]),
                    'temperature': float(parts[5]),
                    'power': float(parts[6])
                }
        
        return None
    except Exception as e:
        print(f"Error getting GPU metrics for GPU {gpu_index}: {e}", file=sys.stderr)
        return None


def update_metrics():
    """Update Prometheus metrics from nvidia-smi"""
    service = os.getenv('SERVICE_NAME', 'demo-service')
    namespace = os.getenv('NAMESPACE', 'default')
    
    gpus = get_gpu_info()
    
    if not gpus:
        print("No GPUs found or nvidia-smi not available", file=sys.stderr)
        return
    
    for gpu in gpus:
        gpu_index = gpu['index']
        gpu_name = gpu['name']
        gpu_type = gpu_name.replace(' ', '_').replace('-', '_')
        
        metrics = get_gpu_metrics(gpu_index, gpu_name)
        
        if metrics:
            # Update metrics
            nvidia_gpu_utilization.labels(
                gpu=f'gpu{gpu_index}',
                gpu_type=gpu_type,
                service=service,
                namespace=namespace
            ).set(metrics['utilization'])
            
            # Memory in bytes
            memory_used_bytes = int(metrics['memory_used_mb'] * 1024 * 1024)
            memory_total_bytes = int(metrics['memory_total_mb'] * 1024 * 1024)
            
            nvidia_gpu_memory_used_bytes.labels(
                gpu=f'gpu{gpu_index}',
                gpu_type=gpu_type,
                service=service,
                namespace=namespace
            ).set(memory_used_bytes)
            
            nvidia_gpu_memory_total_bytes.labels(
                gpu=f'gpu{gpu_index}',
                gpu_type=gpu_type,
                service=service,
                namespace=namespace
            ).set(memory_total_bytes)
            
            nvidia_gpu_temperature.labels(
                gpu=f'gpu{gpu_index}',
                gpu_type=gpu_type,
                service=service,
                namespace=namespace
            ).set(metrics['temperature'])
            
            nvidia_gpu_power_usage.labels(
                gpu=f'gpu{gpu_index}',
                gpu_type=gpu_type,
                service=service,
                namespace=namespace
            ).set(metrics['power'])


def main():
    """Main entry point"""
    port = int(os.getenv('PROMETHEUS_PORT', '9100'))
    
    print(f"Starting GPU metrics exporter on port {port}")
    print(f"Service: {os.getenv('SERVICE_NAME', 'demo-service')}")
    print(f"Namespace: {os.getenv('NAMESPACE', 'default')}")
    
    # Start Prometheus metrics server
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
    
    # Update metrics every 5 seconds
    while True:
        try:
            update_metrics()
        except Exception as e:
            print(f"Error updating metrics: {e}", file=sys.stderr)
        
        time.sleep(5)


if __name__ == '__main__':
    main()

