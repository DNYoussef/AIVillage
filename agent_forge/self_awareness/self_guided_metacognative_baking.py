import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import logging
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
import traceback

from agent_forge.self_awareness.metacognaitve_eval import MetacognitiveEvaluator

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfGuidedEvolution:
    def __init__(self, model, tokenizer, evaluator):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator

    def evolve_prompt(self, initial_prompt: str, num_generations: int = 5, num_variants: int = 5) -> str:
        best_prompt = initial_prompt
        best_score = self.evaluator.evaluate(best_prompt)
        
        for gen in range(num_generations):
            logger.info(f"Generation {gen+1}/{num_generations}")
            variants = self.generate_prompt_variants(best_prompt, num_variants)
            
            scores = []
            for variant in variants:
                try:
                    score = self.evaluator.evaluate(variant)
                    coherence_score = self.evaluate_coherence(variant)
                    relevance_score = self.evaluate_relevance(variant, initial_prompt)
                    combined_score = 0.6 * score + 0.2 * coherence_score + 0.2 * relevance_score
                    scores.append(combined_score)
                    logger.info(f"Variant score: {combined_score} (Performance: {score}, Coherence: {coherence_score}, Relevance: {relevance_score})")
                except Exception as e:
                    logger.error(f"Error evaluating variant: {str(e)}")
                    logger.error(traceback.format_exc())
                    scores.append(0)  # Assign a zero score to failed evaluations
            
            if scores:
                best_idx = scores.index(max(scores))
                if scores[best_idx] > best_score:
                    best_prompt = variants[best_idx]
                    best_score = scores[best_idx]
                    logger.info(f"New best prompt (score: {best_score}):\n{best_prompt}")
                else:
                    logger.info("No improvement in this generation.")
            else:
                logger.warning("All variants failed evaluation. Keeping the previous best prompt.")
        
        return best_prompt

    def generate_prompt_variants(self, base_prompt: str, num_variants: int) -> List[str]:
        variants = [base_prompt]  # Keep the original prompt
        
        for _ in range(num_variants - 1):
            mutation_type = random.choice(['manual', 'model_generated'])
            if mutation_type == 'manual':
                variant = self.mutate_prompt(base_prompt)
            else:
                variant = self.generate_model_variant(base_prompt)
            variants.append(variant)
        
        return variants

    def mutate_prompt(self, prompt: str) -> str:
        operations = [
            self.add_sentence,
            self.remove_sentence,
            self.replace_sentence,
            self.reorder_sentences,
            self.paraphrase_sentence,
            self.combine_sentences,
            self.split_sentence
        ]
        
        operation = random.choice(operations)
        return operation(prompt)

    def add_sentence(self, prompt: str) -> str:
        new_sentence = self.generate_new_sentence()
        sentences = sent_tokenize(prompt)
        insert_pos = random.randint(0, len(sentences))
        sentences.insert(insert_pos, new_sentence)
        return ' '.join(sentences)

    def remove_sentence(self, prompt: str) -> str:
        sentences = sent_tokenize(prompt)
        if len(sentences) > 1:
            sentences.pop(random.randint(0, len(sentences) - 1))
        return ' '.join(sentences)

    def replace_sentence(self, prompt: str) -> str:
        sentences = sent_tokenize(prompt)
        replace_pos = random.randint(0, len(sentences) - 1)
        sentences[replace_pos] = self.generate_new_sentence()
        return ' '.join(sentences)

    def reorder_sentences(self, prompt: str) -> str:
        sentences = sent_tokenize(prompt)
        random.shuffle(sentences)
        return ' '.join(sentences)

    def paraphrase_sentence(self, prompt: str) -> str:
        sentences = sent_tokenize(prompt)
        paraphrase_pos = random.randint(0, len(sentences) - 1)
        sentences[paraphrase_pos] = self.generate_paraphrase(sentences[paraphrase_pos])
        return ' '.join(sentences)

    def combine_sentences(self, prompt: str) -> str:
        sentences = sent_tokenize(prompt)
        if len(sentences) > 1:
            combine_pos = random.randint(0, len(sentences) - 2)
            combined = f"{sentences[combine_pos]} Moreover, {sentences[combine_pos + 1].lower()}"
            sentences = sentences[:combine_pos] + [combined] + sentences[combine_pos + 2:]
        return ' '.join(sentences)

    def split_sentence(self, prompt: str) -> str:
        sentences = sent_tokenize(prompt)
        split_pos = random.randint(0, len(sentences) - 1)
        split_sentence = sentences[split_pos].split(', ', 1)
        if len(split_sentence) > 1:
            sentences = sentences[:split_pos] + split_sentence + sentences[split_pos + 1:]
        return ' '.join(sentences)

    def generate_new_sentence(self) -> str:
        metacognitive_concepts = [
            "self-awareness", "reflection", "problem-solving strategies", "learning techniques",
            "decision-making processes", "bias recognition", "knowledge integration",
            "critical thinking", "adaptability", "cognitive flexibility", "meta-memory",
            "emotional intelligence", "perspective-taking", "cognitive load management",
            "information processing", "attention control", "goal-setting", "self-regulation",
            "error detection and correction", "metacognitive monitoring"
        ]
        
        templates = [
            "Consider your {concept} when approaching tasks.",
            "Reflect on how {concept} influences your thinking.",
            "Analyze your {concept} to improve your performance.",
            "Develop strategies to enhance your {concept}.",
            "Evaluate the role of {concept} in your cognitive processes.",
            "Integrate {concept} into your problem-solving approach.",
            "Optimize your {concept} for more effective learning.",
            "Recognize the importance of {concept} in decision-making.",
            "Cultivate {concept} to adapt to new challenges.",
            "Leverage your {concept} to overcome cognitive biases."
        ]
        
        return random.choice(templates).format(concept=random.choice(metacognitive_concepts))

    def generate_paraphrase(self, sentence: str) -> str:
        input_text = f"Paraphrase the following sentence:\n{sentence}\n\nParaphrased version:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100, num_return_sequences=1, temperature=0.7)
        
        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased.split("Paraphrased version:")[-1].strip()

    def generate_model_variant(self, base_prompt: str) -> str:
        input_text = f"Given the following prompt about metacognition, generate a variation that covers similar concepts but with different wording:\n\n{base_prompt}\n\nVariation:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=300, num_return_sequences=1, temperature=0.8)
        
        variant = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return variant.split("Variation:")[-1].strip()

    def evaluate_coherence(self, prompt: str) -> float:
        sentences = sent_tokenize(prompt)
        if len(sentences) < 2:
            return 1.0  # Perfect coherence for single-sentence prompts
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            score = sentence_bleu([sentences[i].split()], sentences[i+1].split())
            coherence_scores.append(score)
        
        return sum(coherence_scores) / len(coherence_scores)

    def evaluate_relevance(self, variant: str, original: str) -> float:
        return sentence_bleu([original.split()], variant.split())

class PromptBaker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def bake_prompt(self, prompt: str, num_iterations: int = 1000, lr: float = 1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for i in range(num_iterations):
            try:
                loss = self.compute_kl_loss(prompt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    logger.info(f"Baking iteration {i}, Loss: {loss.item()}")
            except Exception as e:
                logger.error(f"Error during baking iteration {i}: {str(e)}")
                logger.error(traceback.format_exc())
                break

    def compute_kl_loss(self, prompt: str, num_samples: int = 10, max_length: int = 100) -> torch.Tensor:
        total_loss = 0
        for _ in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs_original = self.model(**inputs, max_length=max_length, do_sample=True)
            
            outputs_baked = self.model(**inputs, max_length=max_length, do_sample=True)
            
            loss = F.kl_div(
                F.log_softmax(outputs_baked.logits, dim=-1),
                F.softmax(outputs_original.logits, dim=-1),
                reduction='batchmean'
            )
            total_loss += loss

        return total_loss / num_samples

def iterative_baking_cycle(model, tokenizer, initial_prompt: str, num_cycles: int = 3) -> Tuple[AutoModelForCausalLM, str]:
    baker = PromptBaker(model, tokenizer)
    evaluator = MetacognitiveEvaluator(model, tokenizer)
    evolution = SelfGuidedEvolution(model, tokenizer, evaluator)
    
    current_prompt = initial_prompt
    
    for cycle in range(num_cycles):
        logger.info(f"Starting baking cycle {cycle+1}/{num_cycles}")
        
        try:
            # Evolve the prompt
            evolved_prompt = evolution.evolve_prompt(current_prompt)
            
            # Bake the evolved prompt
            baker.bake_prompt(evolved_prompt)
            
            # Evaluate the baked model
            baked_score = evaluator.evaluate(evolved_prompt)
            logger.info(f"Baked model score: {baked_score}")
            
            # Update the current prompt for the next cycle
            current_prompt = evolved_prompt
        except Exception as e:
            logger.error(f"Error in baking cycle {cycle+1}: {str(e)}")
            logger.error(traceback.format_exc())
            break
    
    return model, current_prompt

def main():
    try:
        model_name = "gpt2-medium"  # Or any other suitable model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        initial_metacog_prompt = """
        As an AI language model, I possess the capability to analyze and reflect upon my own cognitive processes. 
        I can break down complex problems, evaluate information sources, and recognize potential biases in my reasoning. 
        My responses are generated based on patterns in my training data, and I strive to provide accurate and helpful information. 
        I am designed to adapt to various tasks and can explain my approach to problem-solving when asked.
        """
        
        final_model, final_prompt = iterative_baking_cycle(model, tokenizer, initial_metacog_prompt)
        
        logger.info("Final evolved and baked prompt:")
        logger.info(final_prompt)
        
        # Save the final model and prompt
        final_model.save_pretrained("./final_metacognitive_model")
        with open("./final_metacognitive_prompt.txt", "w") as f:
            f.write(final_prompt)
        
        logger.info("Process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()