import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from agent_forge.model_compression.model_compression import TernaryQuantizer
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_ternary_compression(model: nn.Module, chunk_size: int = 100000) -> Dict[str, Any]:
    """Apply simple ternary compression to model."""
    try:
        compressed_state = {}
        
        # Convert each parameter to ternary values
        for name, param in model.named_parameters():
            if 'weight' in name:
                logger.info(f"Compressing {name} with shape {param.shape}")
                # Process in chunks to avoid OOM
                weight_flat = param.data.view(-1)
                chunks = torch.split(weight_flat, chunk_size)
                quantized_chunks = []
                scales = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    chunk_quant, chunk_scale = TernaryQuantizer.apply(chunk)
                    quantized_chunks.append(chunk_quant)
                    scales.append(chunk_scale)
                
                # Combine chunks
                quantized = torch.cat(quantized_chunks).view_as(param.data)
                scale = torch.mean(torch.stack(scales))
                
                compressed_state[name] = {
                    'quantized': quantized,
                    'scale': scale
                }
            else:
                compressed_state[name] = param.data
        
        return compressed_state
        
    except Exception as e:
        logger.error(f"Error in simple ternary compression: {str(e)}")
        raise

async def compress_model(model_id: str):
    """Compress a single model."""
    try:
        # Load model and tokenizer
        logger.info(f"Loading model {model_id}")
        
        try:
            # First try loading as CausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True  # Allow custom model code
            )
        except Exception as e:
            if "multi_modality" in str(e):
                # For multi-modal models, use base model
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True  # Allow custom model code
                )
            else:
                raise
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {total_params:,} parameters")
        
        # Choose chunk size based on model size
        if total_params > 1e9:  # >1B params
            chunk_size = 50000
        elif total_params > 1e8:  # >100M params
            chunk_size = 100000
        else:  # Smaller models
            chunk_size = 200000
        
        # Compress model
        logger.info(f"Compressing model with chunk size {chunk_size}")
        compressed_state = simple_ternary_compression(model, chunk_size)
        
        # Calculate compression metrics
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compressed_size = sum(
            v['quantized'].numel() if isinstance(v, dict) else v.numel()
            for v in compressed_state.values()
        )
        compression_ratio = original_size / compressed_size
        
        metrics = {
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'bits_per_parameter': (compressed_size * 8) / total_params
        }
        
        # Save compressed model
        model_name = model_id.split('/')[-1].lower()
        output_dir = f"compressed_{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving compressed model to {output_dir}")
        # Save compressed state
        torch.save(compressed_state, os.path.join(output_dir, "compressed_state.pt"))
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        # Save compression info
        torch.save({
            'compression_metrics': metrics,
            'chunk_size': chunk_size,
            'total_params': total_params
        }, os.path.join(output_dir, "compression_info.pt"))
        
        # Log compression results
        logger.info("Compression complete!")
        logger.info(f"Original size: {metrics['original_size_mb']:.2f} MB")
        logger.info(f"Compressed size: {metrics['compressed_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        logger.info(f"Bits per parameter: {metrics['bits_per_parameter']:.2f}")
        
        # Clean up to free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise

async def main():
    # Models to compress (using only public models)
    models = [
        "Qwen/Qwen1.5-0.5B",  # Smaller Qwen model
        "deepseek-ai/deepseek-coder-1.3b-base",  # Alternative to Janus
        "TinyLlama/TinyLlama-1.1B-Chat-v0.6"  # Correct TinyLlama model ID
    ]
    
    # Compress each model
    for model_id in models:
        logger.info(f"\nCompressing {model_id}")
        try:
            output_dir = await compress_model(model_id)
            logger.info(f"Model compressed and saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to compress {model_id}: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(main())
