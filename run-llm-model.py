#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run LLM Model - Test script to run a small LLM model
This will be detected by LLM Monitor

Usage: python run-llm-model.py [--model-name MODEL_NAME]
Example: python run-llm-model.py --model-name microsoft/DialoGPT-small
"""

import time
import sys
import argparse
import logging
import os
import json
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
parser = argparse.ArgumentParser(description='Run LLM model for testing')
parser.add_argument('--model-name', type=str, default='microsoft/DialoGPT-small',
                    help='Model name to load (e.g., microsoft/DialoGPT-small)')
args = parser.parse_args()

model_name = args.model_name

# Set environment variable so LLM Monitor can detect it automatically
os.environ['MODEL_NAME'] = model_name
os.environ['TRANSFORMERS_MODEL_NAME'] = model_name

logger.info(f"=== LLM Model Service Starting ===")
logger.info(f"Model: {model_name}")
logger.info("This is a small model for testing...")

logger.info(f"Loading tokenizer from {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    sys.exit(1)

logger.info(f"Loading model from {model_name}...")
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    model = model.to(device)
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Model moved to GPU: {gpu_name}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logger.warning("Running on CPU - inference will be slower")

logger.info("=== Model ready for inference ===")
logger.info("Starting inference loop...")
logger.info("(This will run continuously to simulate LLM usage)")

# Simulate inference loop
for i in range(1000):
    try:
        # User input simulation
        user_input = f"Hello, this is test {i+1}"
        logger.info(f"[USER] Request #{i+1}: {user_input}")
        
        # Prepare input
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)
        
        # Generate response
        logger.debug(f"[SYSTEM] Generating response for request #{i+1}...")
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
        logger.info(f"[MODEL] Response #{i+1}: {generated_text}")
        logger.info(f"[SYSTEM] Inference time: {inference_time:.3f}s, Tokens: {len(outputs[0])}")
        
        # Log GPU usage if available
        if device == "cuda":
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_memory_allocated_bytes = torch.cuda.memory_allocated(0)
            logger.debug(f"[SYSTEM] GPU Memory used: {gpu_memory_allocated:.2f} GB")
            
            # Expose GPU memory to LLM Monitor via file
            # LLM Monitor will read this file to get accurate GPU memory usage
            try:
                gpu_info_file = logs_dir / f'gpu-info-{os.getpid()}.json'
                gpu_info = {
                    'pid': os.getpid(),
                    'gpu_memory_allocated_bytes': int(gpu_memory_allocated_bytes),
                    'gpu_memory_reserved_bytes': int(torch.cuda.memory_reserved(0)),
                    'gpu_utilization': 0.0,  # Not available in this script
                    'gpu_index': 0,
                    'timestamp': time.time()
                }
                with open(gpu_info_file, 'w', encoding='utf-8') as f:
                    json.dump(gpu_info, f)
            except Exception as e:
                logger.debug(f"Failed to write GPU info file: {e}")
        
        print(f"[{i+1}] User: {user_input}")
        print(f"[{i+1}] Model: {generated_text[:100]}...")
        print(f"[{i+1}] Time: {inference_time:.3f}s\n")
        
    except Exception as e:
        logger.error(f"[ERROR] Request #{i+1} failed: {e}", exc_info=True)
    
    time.sleep(10)  # Wait 10 seconds between inferences

