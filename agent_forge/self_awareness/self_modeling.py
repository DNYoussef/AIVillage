# self_modeling.py
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
from text_generation import TextGenerator
import wandb

class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, mask_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        # Apply masking
        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()
        for i in range(len(input_ids)):
            if random.random() < self.mask_probability:
                masked_input_ids[i] = self.tokenizer.mask_token_id
                labels[i] = input_ids[i]
            else:
                labels[i] = -100  # Ignore these tokens in loss computation

        return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "labels": labels}

class SelfModelingTrainer:
    def __init__(self, model_path, rag_system):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.text_generator = TextGenerator(self.model, rag_system)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.current_temperature = 0.0

    def generate_texts(self, temp_range, complexity, curriculum_level):
        return self.text_generator.generate_texts(temp_range, complexity, curriculum_level)

    def set_model_temperature(self, temperature):
        self.current_temperature = temperature

    def train_on_texts(self, texts, temperature, optimizer):
        self.model.train()
        self.set_model_temperature(temperature)
        
        dataset = MaskedTextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Training (temp={temperature:.2f})"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def generate_with_temperature(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=self.current_temperature
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def train_cycle(self, optimizer, curriculum_level):
        complexities = [100, 250, 500, 750, 1000]
        complexity = complexities[min(curriculum_level, len(complexities) - 1)]

        delta = 0.01 * curriculum_level
        temp_ranges = [
            (max(0, 0 - delta), 0.05 + delta),
            (0.2 - delta, 0.3 + delta),
            (0.45 - delta, 0.55 + delta),
            (0.7 - delta, 0.8 + delta),
            (min(0.95 - delta, 1), 1)
        ]

        losses = {}

        generated_texts = {temp_range: self.generate_texts(temp_range, complexity, curriculum_level) 
                           for temp_range in temp_ranges}

        # Training sequence
        for temp, texts in zip([0, 1, 0.25, 0.75, 0.5], 
                               [generated_texts[temp_ranges[0]], generated_texts[temp_ranges[4]],
                                generated_texts[temp_ranges[1]], generated_texts[temp_ranges[3]],
                                generated_texts[temp_ranges[2]]]):
            self.set_model_temperature(temp)
            losses[str(temp)] = self.train_on_texts(texts, temp, optimizer)

        return losses

    def train(self, num_cycles=100, num_curriculum_levels=10):
        wandb.init(project="self-modeling", name="temperature_based_training")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for curriculum_level in range(num_curriculum_levels):
            for cycle in range(num_cycles):
                losses = self.train_cycle(optimizer, curriculum_level)
                
                wandb.log({
                    f"loss_temp_{temp}": loss 
                    for temp, loss in losses.items()
                })
                wandb.log({"cycle": cycle, "curriculum_level": curriculum_level})

            # Save the model after each curriculum level
            self.model.save_pretrained(f"self_modeled_level_{curriculum_level}")
            self.tokenizer.save_pretrained(f"self_modeled_level_{curriculum_level}")

        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run self-modeling training")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--rag_system", type=str, required=True, help="Path to the RAG system")
    parser.add_argument("--num_cycles", type=int, default=100, help="Number of cycles per curriculum level")
    parser.add_argument("--num_curriculum_levels", type=int, default=10, help="Number of curriculum levels")
    args = parser.parse_args()

    trainer = SelfModelingTrainer(args.model_path, args.rag_system)
    trainer.train(num_cycles=args.num_cycles, num_curriculum_levels=args.num_curriculum_levels)