from transformers import AutoModelForCausalLM, AutoTokenizer


class QuietSTAR:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(
        self, prompt: str, max_length: int = 256, thought_tokens: bool = True
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            do_sample=False,
            thought_tokens=thought_tokens,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return text
