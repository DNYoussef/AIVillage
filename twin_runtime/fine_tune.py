"""Nightly on-device fine-tuning using local data."""
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from llama_cpp import Llama
from ingestion.vector_ds import personal_ds

MODEL_PATH = Path(os.getenv("TWIN_MODEL", "~/ai_twin/weights/twin.gguf")).expanduser()
LORA_DIR = Path(os.getenv("TWIN_LORA_DIR", "~/ai_twin/loras")).expanduser()
TOKENIZER_NAME = os.getenv("TWIN_TOKENIZER", "hf-internal-testing/llama-tokenizer")


def run_nightly(user_id: str, steps: int = 2000):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    base = Llama(model_path=str(MODEL_PATH), n_ctx=2048)
    peft_cfg = LoraConfig(r=16, target_modules=["q_proj", "v_proj"], lora_alpha=32)
    model = get_peft_model(base, peft_cfg)

    ds = personal_ds(user_id, max_tokens=steps)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for _ in range(1):
        for batch in loader:
            loss = model(batch)
            loss["loss"].backward()
            opt.step()
            opt.zero_grad()

    out_dir = LORA_DIR / user_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir
