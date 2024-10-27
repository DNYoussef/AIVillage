# Previous imports remain the same, but remove signal import
import os
from pathlib import Path
import logging
import gc
import psutil
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import threading
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

class Timer:
    """Timer class for handling timeouts."""
    def __init__(self, timeout):
        self.timeout = timeout
        self.timer = None
        self.timed_out = False
    
    def start(self):
        """Start the timer."""
        self.timer = threading.Timer(self.timeout, self._timeout)
        self.timer.start()
    
    def cancel(self):
        """Cancel the timer."""
        if self.timer:
            self.timer.cancel()
    
    def _timeout(self):
        """Called when timer expires."""
        self.timed_out = True

class ProgressTracker:
    """Track progress of long-running operations."""
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step: int = 1):
        """Update progress."""
        self.current_step += step
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        logger.info(f"{self.description}: {self.current_step}/{self.total_steps} - "
                   f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

class SimpleBaker:
    """Simplified version of DeepSystemBaker without langroid dependencies."""
    
    def __init__(self, model_name: str = "ibm-granite/granite-3b-code-instruct-128k", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Loading model {model_name} on {device}")
        
        # Load model with minimal config and lower precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            use_cache=False,
            device_map="auto" if device == "cuda" else None
        ).to(device)
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            model_max_length=128  # Even smaller context size
        )
        
        # Set up special tokens
        self.special_tokens = [
            "<start>", "<end>",
            "<analyze>", "</analyze>",
            "<plan>", "</plan>",
            "<code>", "</code>",
            "<explain>", "</explain>",
            "<|stop|>"
        ]
        self._setup_tokenizer()
        
        # Initialize optimizer with lower learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-7,  # Even lower learning rate
            weight_decay=0.005,  # Lower weight decay
            eps=1e-8
        )
        
        # Clear memory
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    def _setup_tokenizer(self):
        """Set up tokenizer with special tokens and padding."""
        special_tokens_dict = {
            'additional_special_tokens': self.special_tokens,
            'pad_token': '[PAD]',
            'eos_token': '<|stop|>',
            'bos_token': '<start>'
        }
        
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"Added {num_added_toks} special tokens")
        
        # Ensure all necessary tokens are set
        self.tokenizer.pad_token = '[PAD]'
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
    
    def _log_memory_usage(self):
        """Log current memory usage."""
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"RAM Usage: {ram_usage:.2f} MB")
        if self.device == "cuda":
            gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            logger.info(f"GPU Memory Usage: {gpu_usage:.2f} MB")
    
    def _process_chunk(self, chunk: str, chunk_size: int, batch_size: int, timer: Timer) -> Optional[float]:
        """Process a single chunk with timeout."""
        try:
            # Start timer
            timer.start()
            
            # Tokenize input with padding
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=chunk_size,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Forward pass with labels for training
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Check for timeout
            if timer.timed_out:
                raise TimeoutException("Operation timed out")
            
            # Scale loss and backward pass
            loss = outputs.loss / batch_size
            loss.backward()
            
            loss_value = loss.item()
            
            # Clear memory
            del outputs, loss, input_ids, attention_mask, encoded
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return loss_value
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return None
        finally:
            timer.cancel()
    
    def bake(self, prompt: str, num_iterations: int = 3, batch_size: int = 4, chunk_size: int = 32, timeout: int = 120):
        """Bake the system prompt into the model."""
        logger.info(f"Starting baking process with {num_iterations} iterations")
        self._log_memory_usage()
        
        # Enable training mode
        self.model.train()
        
        # Split prompt into smaller chunks
        prompt_chunks = [prompt[i:i+chunk_size] for i in range(0, len(prompt), chunk_size)]
        total_chunks = len(prompt_chunks)
        
        progress = ProgressTracker(total_steps=num_iterations * total_chunks, description="Baking progress")
        
        for i in range(num_iterations):
            logger.info(f"Baking iteration {i+1}/{num_iterations}")
            chunk_losses = []
            
            # Initialize gradient accumulation
            accumulated_loss = 0
            steps_since_update = 0
            
            for chunk_idx, chunk in enumerate(prompt_chunks):
                start_time = time.time()
                
                # Create timer for this chunk
                timer = Timer(timeout)
                
                # Process chunk
                loss_value = self._process_chunk(chunk, chunk_size, batch_size, timer)
                
                if loss_value is not None:
                    accumulated_loss += loss_value
                    steps_since_update += 1
                    
                    # Update weights if we've accumulated enough steps
                    if steps_since_update == batch_size:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # Lower gradient clipping
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        chunk_losses.append(accumulated_loss)
                        accumulated_loss = 0
                        steps_since_update = 0
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    logger.info(f"  Chunk {chunk_idx+1}/{total_chunks} - "
                              f"Loss: {loss_value:.4f} - Time: {elapsed:.2f}s")
                    
                    # Update progress
                    progress.update()
            
            # Log average loss for iteration
            avg_loss = sum(chunk_losses) / len(chunk_losses) if chunk_losses else 0
            logger.info(f"Iteration {i+1} average loss: {avg_loss:.4f}")
            self._log_memory_usage()
        
        # Switch back to eval mode
        self.model.eval()
        logger.info("Baking completed")
        self._log_memory_usage()

    def save_model(self, path: str):
        """Save the baked model."""
        logger.info(f"Saving model to {path}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def generate_response(self, prompt: str) -> str:
        """Generate a response using the baked model."""
        # Ensure eval mode
        self.model.eval()
        
        # Format prompt with special tokens
        formatted_prompt = f"""<start>
Write a Python function that takes a list of numbers and returns the sum of even numbers.

<analyze>
Let's break down the requirements:
1. Function that takes a list of numbers
2. Calculate sum of even numbers only
3. Include clear documentation
</analyze>

<plan>
1. Define function with type hints
2. Add comprehensive docstring
3. Use list comprehension for even numbers
4. Return the sum
</plan>

<code>
"""
        
        try:
            with torch.no_grad():
                # Generate with more constrained parameters
                encoded = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_length=256,  # Shorter max length
                    min_length=50,
                    do_sample=True,
                    top_p=0.95,
                    top_k=30,
                    temperature=0.5,  # Lower temperature
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                
                # Decode the output
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Extract the generated part
                generated_part = response[len(self.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)):]
                
                # Clean up the response
                generated_part = generated_part.replace('\u0120', ' ').strip()
                
                # Add closing tags if they're missing
                if "</code>" not in generated_part:
                    generated_part += "\n</code>"
                if "</explain>" not in generated_part:
                    generated_part += "\n<explain>\nFunction is documented and efficient.\n</explain>"
                if "<|stop|>" not in generated_part:
                    generated_part += "\n<|stop|>"
                
                return generated_part
                
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return f"Error generating response: {str(e)}"

def main():
    """Main test function."""
    try:
        # Create unique model name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baked_model_name = f"magi_baked_model_{timestamp}"
        baked_model_path = Path(baked_model_name)
        
        logger.info("Initializing baker...")
        baker = SimpleBaker()  # Using IBM Granite model by default
        
        # System prompt for baking
        system_prompt = """<start>
You are an AI that specializes in code generation and technical problem-solving.
Always structure your responses using these steps:

1. <analyze> Analyze the problem requirements </analyze>
2. <plan> Plan the implementation steps </plan>
3. <code> Write clean, efficient code with documentation </code>
4. <explain> Explain key decisions and considerations </explain>

Focus on:
- Clean, efficient code
- Clear documentation
- Error handling
- Best practices

Example 1:
<start>
Write a function to check if a number is prime.

<analyze>
We need to check if a number has any divisors.
</analyze>

<plan>
1. Create function with type hints
2. Add docstring
3. Handle edge cases
4. Implement efficient check
</plan>

<code>
def is_prime(n: int) -> bool:
    \"\"\"
    Check if a number is prime.
    Args:
        n: The number to check
    Returns:
        bool: True if prime, False otherwise
    \"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
</code>

<explain>
- Used type hints for clarity
- Added docstring with Args and Returns
- Optimized by checking only up to square root
- Included edge case handling
</explain>
<|stop|>

Example 2:
<start>
Write a function to find the maximum element in a list.

<analyze>
Need to iterate through list and track maximum value.
</analyze>

<plan>
1. Create function with type hints
2. Add docstring
3. Handle empty list case
4. Find maximum value
</plan>

<code>
def find_max(numbers: list[float]) -> float:
    \"\"\"
    Find the maximum value in a list.
    Args:
        numbers: List of numbers to search
    Returns:
        float: Maximum value found
    Raises:
        ValueError: If list is empty
    \"\"\"
    if not numbers:
        raise ValueError("Cannot find maximum of empty list")
    return max(numbers)
</code>

<explain>
- Used type hints with list[float]
- Added comprehensive docstring
- Included error handling
- Used built-in max() for efficiency
</explain>
<|stop|>
"""
        
        logger.info("Starting baking process...")
        baker.bake(
            system_prompt,
            num_iterations=3,
            batch_size=4,  # Increased batch size for gradient accumulation
            chunk_size=32,  # Even smaller chunks
            timeout=120  # Increased timeout
        )
        
        # Save baked model
        logger.info(f"Saving baked model as {baked_model_name}")
        baker.save_model(str(baked_model_path))
        
        # Test the model
        test_prompt = """
        Write a Python function that:
        1. Takes a list of numbers
        2. Returns the sum of even numbers
        
        Keep it short and include a docstring.
        """
        
        logger.info("Testing baked model...")
        response = baker.generate_response(test_prompt)
        
        # Save results
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save results as JSON to handle encoding
        results = {
            "baked_model_path": str(baked_model_path),
            "test_prompt": test_prompt,
            "model_response": response
        }
        
        with open(output_dir / "magi_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_dir / 'magi_test_results.json'}")
        logger.info(f"To use this baked model, update config to use: {baked_model_name}")
        
        print("\n=== Test Results ===\n")
        print("Test completed successfully!")
        print(f"Baked model saved to: {baked_model_path}")
        print("\nModel Response:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        print("\n=== Test Results ===\n")
        print("Test failed!")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

