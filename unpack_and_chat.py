from agent_forge.model_compression import FinalCompressor
from agent_forge.model_compression.inference_engine import InferenceEngine
from transformers import AutoTokenizer
import torch

def main():
    # 1. Compress the model
    compressor = FinalCompressor()
    model = torch.load("qwen_model.pt")
    compressed_state = compressor.compress_model(model)
    torch.save(compressed_state, "compressed_model.pt")
    
    # 2. Decompress and interact
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    engine = InferenceEngine(torch.load("compressed_model.pt"))
    
    print("System ready. Type 'exit' to quit.")
    while True:
        prompt = input("\nUser: ")
        if prompt.lower() == "exit":
            break
            
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = engine.generate(inputs, max_length=100, temperature=0.7)
        print(f"AI: {tokenizer.decode(outputs[0])}")

if __name__ == "__main__":
    main()