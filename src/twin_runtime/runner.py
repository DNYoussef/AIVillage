import json
import os
from pathlib import Path
import logging

from chromadb import PersistentClient
from llama_cpp import Llama
import peft

logger = logging.getLogger(__name__)

try:
    from transformers import LlamaConfig, LlamaForCausalLM
except Exception:  # transformers optional in test env
    LlamaConfig = None
    LlamaForCausalLM = None

_MODEL = Path(os.getenv("TWIN_MODEL", "~/ai_twin/weights/twin.gguf")).expanduser()
_LORA_DIR = Path(os.getenv("TWIN_LORA_DIR", "~/ai_twin/loras")).expanduser()
_HOME = Path(os.getenv("TWIN_HOME", "~/.twin_chroma")).expanduser()
_COMPRESSED = Path(os.getenv("TWIN_COMPRESSED", "")).expanduser()

DB = PersistentClient(path=_HOME)


def _load_llm():
    if _COMPRESSED.exists() and LlamaConfig is not None:
        try:
            from twin_runtime.compressed_loader import CompressedModelLoader

            def _empty_llama():
                return LlamaForCausalLM(LlamaConfig())

            loader = CompressedModelLoader(_empty_llama, str(_COMPRESSED))
            return loader.assemble_model()
        except Exception as e:  # noqa: PERF203
            logger.error("Failed to load compressed model from %s: %s", _COMPRESSED, e)
    return Llama(
        model_path=str(_MODEL), n_ctx=4096, n_threads=max(os.cpu_count() // 2, 1)
    )


LLM = _load_llm()


def _merge_domain_lora(prompt: str):
    if "finance" in prompt.lower() and (_LORA_DIR / "finance").exists():
        peft.LoraModel.from_pretrained(LLM, _LORA_DIR / "finance").merge_and_unload()


def chat(prompt: str, **kw):
    _merge_domain_lora(prompt)
    coll = DB.get_or_create_collection(f"user:{kw.get('user_id', 'default')}")
    ctx = coll.query(query_texts=[prompt], n_results=3)
    context = "\n".join(ctx.get("documents", [[]])[0])
    full_prompt = f"<CTX>{context}</CTX>\n\n{prompt}"
    out = LLM(full_prompt, temperature=0.4, max_tokens=kw.get("max_tokens", 256))
    ans = out["choices"][0]["text"]
    if kw.get("log"):
        path = _HOME / "chat.log"
        with open(path, "a", encoding="utf8") as fh:
            fh.write(json.dumps({"q": prompt, "a": ans}) + "\n")
    return ans
