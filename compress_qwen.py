import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent_forge.model_compression.model_compression import (
    CompressionConfig,
    CompressedModel,
    compress_and_train
)
from agent_forge.model_compression.hypercompression import (
    FinalCompressionConfig,
    FinalCompressor,
    CompressionBenchmark
)
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompressedQwenModel(CompressedModel):
    """Custom CompressedModel for Qwen that handles dictionary inputs."""
    def forward(self, x):
        # Extract input_ids from dictionary input
        if isinstance(x, dict):
            return self.model(**x)
        return self.model(x)

def create_training_data(tokenizer, max_length=128):
    """Create training data with proper padding."""
    train_prompts = [
        "Calculate the derivative of x^2",
        "Solve the equation 2x + 5 = 13",
        "Find the area of a circle with radius 5",
        "What is the integral of sin(x)?",
        "Calculate the limit of 1/x as x approaches infinity",
        "Find the eigenvalues of matrix [[1,2],[3,4]]",
        "Solve the differential equation dy/dx = 2x",
        "Calculate the variance of dataset [1,2,3,4,5]"
    ]
    
    # Tokenize all prompts with padding
    encodings = tokenizer(
        train_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create training pairs
    train_data = []
    for i in range(len(train_prompts)):
        # Get input_ids and attention_mask for this example
        input_ids = encodings['input_ids'][i]
        attention_mask = encodings['attention_mask'][i]
        
        # Create a simple target (shift input by 1 position)
        target = input_ids.clone()
        target[:-1] = input_ids[1:]
        target[-1] = tokenizer.pad_token_id
        
        # Create data dictionary
        data = {
            'input_ids': input_ids.unsqueeze(0),  # Add batch dimension
            'attention_mask': attention_mask.unsqueeze(0)  # Add batch dimension
        }
        
        train_data.append((data, target))
    
    return train_data

class MathDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function to handle batched data."""
    # Separate data and targets
    data_list = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Combine input_ids and attention_masks
    input_ids = torch.cat([d['input_ids'] for d in data_list], dim=0)
    attention_masks = torch.cat([d['attention_mask'] for d in data_list], dim=0)
    
    # Stack targets
    targets = torch.stack(targets)
    
    # Create batched data dictionary
    batched_data = {
        'input_ids': input_ids,
        'attention_mask': attention_masks
    }
    
    return batched_data, targets

async def main():
    # Configure compression settings
    initial_config = CompressionConfig(
        # VPTQ settings - aggressive compression
        vector_size=4,  # Smaller vectors for more compression
        codebook_size=128,  # Smaller codebook
        group_size=64,  # Smaller groups
        
        # BitNet settings
        lambda_warmup=500,  # Faster warmup for testing
        lambda_schedule='linear',
        
        # Training settings
        batch_size=4,  # Smaller batch size due to model size
        learning_rate=1e-5,  # Conservative learning rate
        epochs=1,  # Start with 1 epoch for testing
        
        # Hardware settings
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True,
        num_workers=4
    )

    # Stage 3 & 4: HyperCompression + SeedLM Configuration
    final_config = FinalCompressionConfig(
        # HyperCompression params
        block_size=128,  # Smaller blocks for more compression
        theta_max=500000,  # Reduced from default
        chunk_size=500000,  # Reduced from default
        
        # SeedLM params
        lfsr_length=12,  # Reduced from default 16
        lfsr_polynomial=0x1100B,
        
        # General params
        num_threads=4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_mixed_precision=True
    )

    # Load model and tokenizer
    model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
    logger.info(f"Loading model {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create training dataset
    logger.info("Preparing training data")
    train_data = create_training_data(tokenizer)
    dataset = MathDataset(train_data)
    
    # Create data loaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=initial_config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=initial_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    try:
        # Stage 1 & 2: VPTQ + BitNet compression
        logger.info("Starting VPTQ + BitNet compression")
        # Use custom CompressedQwenModel instead of CompressedModel
        compressed_model = CompressedQwenModel(model, initial_config)
        compressed_model, stats = await compress_and_train(
            model=compressed_model,  # Pass already wrapped model
            train_loader=train_loader,
            val_loader=val_loader,
            config=initial_config
        )
        
        # Stage 3 & 4: HyperCompression + SeedLM
        logger.info("Starting HyperCompression + SeedLM compression")
        final_compressor = FinalCompressor(final_config)
        compressed_state = final_compressor.compress_model(compressed_model)
        
        # Calculate compression metrics
        metrics = CompressionBenchmark.calculate_metrics(model, compressed_state)
        
        # Save compressed model
        output_dir = "compressed_qwen_math"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving compressed model to {output_dir}")
        # Save compressed state
        torch.save(compressed_state, os.path.join(output_dir, "compressed_state.pt"))
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        # Save compression configs
        torch.save({
            'initial_config': initial_config,
            'final_config': final_config,
            'compression_metrics': metrics
        }, os.path.join(output_dir, "compression_config.pt"))
        
        # Log compression results
        logger.info("Compression complete!")
        logger.info(f"Original size: {metrics['original_size_mb']:.2f} MB")
        logger.info(f"Compressed size: {metrics['compressed_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        logger.info(f"Bits per parameter: {metrics['bits_per_parameter']:.2f}")
        logger.info(f"Initial compression loss: {stats['final_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
