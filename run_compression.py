import asyncio
import logging
from initialize_village import AIVillage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_compression_task():
    """Run the model compression task using the initialized AI Village."""
    try:
        # Initialize AI Village
        village = AIVillage()
        await village.initialize()
        
        # Create compression task
        compression_task = {
            "type": "model_compression",
            "content": {
                "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
                "compression_stages": [
                    "VPTQ Compression",
                    "BitNet Quantization",
                    "HyperCompression",
                    "SeedLM Compression"
                ],
                "requirements": {
                    "maintain_math_accuracy": True,
                    "target_size_reduction": "6x",
                    "preserve_capabilities": [
                        "arithmetic",
                        "algebra",
                        "calculus"
                    ]
                }
            },
            "priority": "high"
        }
        
        # Process compression task
        logger.info("Starting model compression task...")
        result = await village.process_task(compression_task)
        
        # Log results
        logger.info("Compression task complete!")
        logger.info(f"Result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running compression task: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        result = asyncio.run(run_compression_task())
        logger.info("Compression task execution complete.")
    except Exception as e:
        logger.error(f"Failed to run compression task: {str(e)}")
        raise
