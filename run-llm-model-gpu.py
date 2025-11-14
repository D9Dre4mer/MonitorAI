#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run LLM Model with GPU - Test script to run LLM model with GPU acceleration
This will be detected by LLM Monitor and show GPU metrics

Usage: python run-llm-model-gpu.py [--model-name MODEL_NAME]
Example: python run-llm-model-gpu.py --model-name gpt2
"""

import time
import sys
import argparse
import logging
import os
import json
from datetime import datetime
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Installing transformers and torch...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "torch"])
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

# Setup logging directory
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / 'llm-model.log'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run LLM model with GPU for testing')
parser.add_argument('--model-name', type=str, default='gpt2',
                    help='Model name to load (e.g., gpt2, distilgpt2, microsoft/phi-2)')
parser.add_argument('--force-cpu', action='store_true',
                    help='Force CPU usage even if GPU is available')
args = parser.parse_args()

model_name = args.model_name

# Set environment variable so LLM Monitor can detect it automatically
os.environ['MODEL_NAME'] = model_name
os.environ['TRANSFORMERS_MODEL_NAME'] = model_name

logger.info(f"=== LLM Model Service Starting (GPU Mode) ===")
logger.info(f"Model: {model_name}")

# Check GPU availability
if args.force_cpu:
    device = "cpu"
    logger.warning("Forcing CPU usage (--force-cpu flag)")
else:
    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Available: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {gpu_count}")
    else:
        device = "cpu"
        logger.warning("GPU not available - falling back to CPU")
        logger.warning("Install CUDA-enabled PyTorch for GPU acceleration")

logger.info(f"Using device: {device}")

logger.info(f"Loading tokenizer from {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    sys.exit(1)

logger.info(f"Loading model from {model_name}...")
try:
    # Load model with device_map for automatic GPU placement
    if device == "cuda":
        logger.info("Loading model to GPU...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for GPU
                device_map="auto"  # Automatically place on GPU
            )
        except Exception as e:
            logger.warning(f"device_map='auto' failed: {e}, trying manual GPU placement...")
            # Fallback: load to CPU then move to GPU
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device)
            model = model.half()  # Convert to float16
        logger.info("Model loaded successfully")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Model loaded on CPU")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# Verify GPU usage
if device == "cuda":
    # Check if model is actually on GPU
    model_device = next(model.parameters()).device
    if model_device.type == "cuda":
        logger.info(f"✓ Model is on GPU: {model_device}")
        initial_memory = torch.cuda.memory_allocated(0) / 1024**3
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"✓ GPU Memory - Allocated: {initial_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB")
        
        # Force GPU usage by doing a dummy forward pass
        logger.info("Performing dummy forward pass to ensure GPU is active...")
        dummy_input = tokenizer.encode("test", return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(dummy_input, max_length=5, do_sample=False)
        logger.info("✓ Dummy forward pass completed - GPU should now be visible to nvidia-smi")
    else:
        logger.error(f"✗ Model is on {model_device}, not GPU! Check CUDA installation.")
        logger.error("  Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        device = "cpu"
else:
    logger.info("Model is on CPU")

logger.info("=== Model ready for inference ===")
logger.info("Starting inference loop with GPU acceleration...")
logger.info("(This will run continuously to simulate LLM usage)")

# Simulate inference loop
iteration = 0
try:
    while True:
        iteration += 1
        try:
            # User input simulation
            user_input = f"Hello, this is GPU test {iteration}"
            logger.info(f"[USER] Request #{iteration}: {user_input}")
            
            # Prepare input
            inputs = tokenizer.encode(user_input, return_tensors="pt")
            if device == "cuda":
                inputs = inputs.to(device)
            
            # Generate response
            logger.debug(f"[SYSTEM] Generating response for request #{iteration}...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs, 
                    max_length=50, 
                    do_sample=True, 
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    top_p=0.9
                )
            
            inference_time = time.time() - start_time
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Log model response
            logger.info(f"[MODEL] Response #{iteration}: {generated_text}")
            logger.info(f"[SYSTEM] Inference time: {inference_time:.3f}s, Tokens: {len(outputs[0])}")
            
            # Log GPU usage if available
            if device == "cuda":
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                gpu_memory_allocated_bytes = torch.cuda.memory_allocated(0)
                
                # Try to get GPU utilization, but handle if pynvml is not installed
                gpu_utilization = 0.0
                try:
                    if hasattr(torch.cuda, 'utilization'):
                        gpu_utilization = torch.cuda.utilization()
                        logger.info(f"[SYSTEM] GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")
                        logger.info(f"[SYSTEM] GPU Utilization: {gpu_utilization}%")
                    else:
                        logger.info(f"[SYSTEM] GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")
                except (ModuleNotFoundError, AttributeError) as e:
                    # pynvml not installed or utilization not available
                    logger.info(f"[SYSTEM] GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")
                    logger.debug(f"[SYSTEM] GPU Utilization not available (pynvml may not be installed): {e}")
                
                # Expose GPU memory to LLM Monitor via file
                # LLM Monitor will read this file to get accurate GPU memory usage
                try:
                    import json
                    gpu_info_file = Path('logs') / f'gpu-info-{os.getpid()}.json'
                    gpu_info = {
                        'pid': os.getpid(),
                        'gpu_memory_allocated_bytes': int(gpu_memory_allocated_bytes),
                        'gpu_memory_reserved_bytes': int(torch.cuda.memory_reserved(0)),
                        'gpu_utilization': float(gpu_utilization),
                        'gpu_index': 0,
                        'timestamp': time.time()
                    }
                    with open(gpu_info_file, 'w', encoding='utf-8') as f:
                        json.dump(gpu_info, f)
                except Exception as e:
                    logger.debug(f"Failed to write GPU info file: {e}")
            
            print(f"[{iteration}] User: {user_input}")
            print(f"[{iteration}] Model: {generated_text[:100]}...")
            print(f"[{iteration}] Time: {inference_time:.3f}s")
            if device == "cuda":
                print(f"[{iteration}] GPU Memory: {gpu_memory_allocated:.2f} GB\n")
            else:
                print()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"[ERROR] Request #{iteration} failed: {e}", exc_info=True)
        
        time.sleep(10)  # Wait 10 seconds between inferences

except KeyboardInterrupt:
    logger.info("Shutting down gracefully...")
finally:
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

