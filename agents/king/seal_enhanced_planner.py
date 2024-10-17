from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict, Any
from .plan_generator import PlanGenerator  # Assuming this is your existing reverse tree planner

class SEALEnhancedPlanGenerator:
    def __init__(self, model_name='gpt2'):
        self.reverse_tree_planner = PlanGenerator()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    async def generate_enhanced_plan(self, task: str, rag_info: Dict[str, Any]) -> Dict[str, Any]:
        # First, generate the reverse tree plan
        initial_plan = await self.reverse_tree_planner.generate_plan(task, rag_info)
        
        # Now, enhance each sub-goal in the plan
        enhanced_plan = await self._enhance_plan(initial_plan, rag_info)
        
        return enhanced_plan

    async def _enhance_plan(self, plan: Dict[str, Any], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_plan = {}
        for key, value in plan.items():
            if isinstance(value, dict):
                # This is a sub-goal, enhance it
                enhanced_sub_goal = await self._enhance_sub_goal(key, value, rag_info)
                enhanced_plan[key] = await self._enhance_plan(enhanced_sub_goal, rag_info)
            else:
                # This is a leaf node (action), keep it as is
                enhanced_plan[key] = value
        return enhanced_plan

    async def _enhance_sub_goal(self, sub_goal: str, sub_plan: Dict[str, Any], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Task: {sub_goal}\nExisting Plan: {sub_plan}\nAdditional Info: {rag_info}\nEnhance and expand this sub-goal:"
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        enhanced_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse the enhanced text into a dictionary structure
        # This is a simplified parsing, you might need a more sophisticated parser
        enhanced_lines = enhanced_text.split('\n')
        enhanced_sub_plan = {}
        for line in enhanced_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                enhanced_sub_plan[key.strip()] = value.strip()
        
        # Merge the enhanced sub-plan with the original sub-plan
        merged_sub_plan = {**sub_plan, **enhanced_sub_plan}
        
        return merged_sub_plan

    async def update(self, task: Dict, result: Dict):
        # Update both the reverse tree planner and the language model
        await self.reverse_tree_planner.update(task, result)
        # Fine-tune the language model based on task execution results
        # This is a placeholder; actual implementation would involve more complex fine-tuning
        pass

    def save(self, path: str):
        self.model.save_pretrained(f"{path}/seal_model")
        self.tokenizer.save_pretrained(f"{path}/seal_tokenizer")
        # Add logic to save reverse_tree_planner if needed

    def load(self, path: str):
        self.model = GPT2LMHeadModel.from_pretrained(f"{path}/seal_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(f"{path}/seal_tokenizer")
        self.model.to(self.device)
        # Add logic to load reverse_tree_planner if needed
