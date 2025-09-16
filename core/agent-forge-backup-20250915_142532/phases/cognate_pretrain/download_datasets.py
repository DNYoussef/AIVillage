#!/usr/bin/env python3
"""
Download Real Datasets for Cognate Pretraining

Downloads and prepares the exact datasets specified:
- Short/local: GSM8K, SVAMP, ASDiv, Mini-MBPP/CodeXGLUE edits, short infill
- Long-horizon: HotpotQA, 2WikiMultiHopQA, MuSiQue, QuALITY, Qasper, NarrativeQA
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CognateDatasetDownloader:
    """Downloads and prepares datasets for Cognate pretraining curriculum."""

    def __init__(self, output_dir: str = "./cognate_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dataset configurations
        self.short_datasets = {
            "GSM8K": "gsm8k",
            "SVAMP": "ChilleD/SVAMP",
            "ASDiv": "EleutherAI/asdiv",
            "Mini-MBPP": "mbpp",
            "CodeXGLUE": "microsoft/CodeXGLUE-code-completion-line",
        }

        self.long_datasets = {
            "HotpotQA": "hotpot_qa",
            "2WikiMultiHopQA": "2wikimultihopqa",
            "MuSiQue": "musique",
            "QuALITY": "quality",
            "Qasper": "qasper",
            "NarrativeQA": "narrativeqa",
        }

    def download_dataset(self, dataset_name: str, hf_name: str, split_type: str = "short") -> bool:
        """Download a single dataset from HuggingFace with robust fallbacks and safe decoding."""
        import re

        try:
            logger.info(f"Downloading {dataset_name} ({hf_name})...")
            dataset_dir = self.output_dir / dataset_name.lower()
            dataset_dir.mkdir(exist_ok=True)
            processed_data = []

            # Preferred dataset config overrides for multi-config datasets
            preferred_config = None
            if dataset_name == "HotpotQA":
                preferred_config = "fullwiki"

            load_kwargs = {"split": "train", "trust_remote_code": True}
            dataset = None

            # Primary load attempt (with preferred config when known)
            try:
                if preferred_config:
                    dataset = load_dataset(hf_name, preferred_config, **load_kwargs)
                else:
                    dataset = load_dataset(hf_name, **load_kwargs)
            except Exception as e_load:
                err_msg = str(e_load)
                logger.warning(f"Initial load failed for {hf_name}: {err_msg}")

                # If the dataset requires a config, try to parse available configs and pick one
                if "Please pick one among the available configs" in err_msg or "Config name is missing" in err_msg:
                    configs = re.findall(r"'([^']+)'", err_msg)
                    if configs:
                        chosen = "fullwiki" if "fullwiki" in configs else configs[0]
                        logger.info(f"Selecting dataset config '{chosen}' for {hf_name} based on available configs")
                        dataset = load_dataset(hf_name, chosen, **load_kwargs)
                    else:
                        logger.error(f"Could not parse dataset configs from error: {err_msg}")
                        raise

                # If the dataset doesn't exist or other fatal error, try streaming as a last resort
                elif "doesn't exist on the Hub" in err_msg or "NotFoundError" in err_msg:
                    logger.error(f"Dataset {hf_name} not found on the Hub: {err_msg}")
                    return False
                else:
                    # Try streaming mode to avoid some decoding issues or large-file problems
                    try:
                        logger.info(f"Attempting streaming load for {hf_name} as a fallback")
                        dataset = load_dataset(hf_name, streaming=True)
                    except Exception as e_stream:
                        logger.error(f"Streaming fallback also failed for {hf_name}: {e_stream}")
                        raise

            # Helper to safely convert various field types to short text
            def _safe_text(val, max_len=500):
                try:
                    if val is None:
                        return ""
                    if isinstance(val, bytes | bytearray):
                        return val.decode("utf-8", errors="replace")[:max_len]
                    if isinstance(val, list):
                        return " ".join(str(x) for x in val)[:max_len]
                    return str(val)[:max_len]
                except Exception:
                    return repr(val)[:max_len]

            # Processing rules per-dataset (defensive: use .get and safe conversions)
            if dataset_name == "GSM8K":
                for item in dataset:
                    q = item.get("question") or item.get("question_text") or item.get("problem")
                    a = item.get("answer") or item.get("answer_text") or item.get("correct_answer")
                    if q and a:
                        processed_data.append(
                            {
                                "text": f"Problem: {_safe_text(q,300)} Solution: {_safe_text(a,300)}",
                                "seq_type": "short",
                                "dataset": "GSM8K",
                                "requires_memory": False,
                                "metadata": {"domain": "math", "complexity": "grade_school"},
                            }
                        )

            elif dataset_name == "HotpotQA":
                # Items can differ depending on config; handle both dict/list contexts
                for item in dataset:
                    try:
                        question = item.get("question") or item.get("query") or item.get("question_text", "")
                        # context can be dict with 'sentences' or list of docs
                        context_text = ""
                        ctx = item.get("context") or item.get("context_docs") or item.get("context_paragraphs")
                        if isinstance(ctx, dict) and "sentences" in ctx:
                            context_text = " ".join([_safe_text(s, 200) for s in ctx["sentences"][:5]])
                        elif isinstance(ctx, list):
                            # If each entry is a dict or string
                            snippets = []
                            for c in ctx[:5]:
                                if isinstance(c, dict):
                                    snippets.append(_safe_text(c.get("text") or c.get("content") or c, 200))
                                else:
                                    snippets.append(_safe_text(c, 200))
                            context_text = " ".join(snippets)
                        answer = item.get("answer") or item.get("answers") or item.get("final_answer") or ""
                        processed_data.append(
                            {
                                "text": f"Context: {context_text} Question: {_safe_text(question,300)} Answer: {_safe_text(answer,300)}",
                                "seq_type": "long",
                                "dataset": "HotpotQA",
                                "requires_memory": True,
                                "metadata": {"hops": 2, "reasoning_type": "multi_hop"},
                            }
                        )
                    except Exception:
                        # If a single item fails, skip it but continue
                        logger.debug("Skipping a malformed HotpotQA item", exc_info=True)
                        continue

            elif dataset_name == "SVAMP":
                for item in dataset:
                    body = item.get("Body") or item.get("body") or ""
                    question = item.get("Question") or item.get("question") or ""
                    answer = item.get("Answer") or item.get("answer") or ""
                    text = (
                        f"Problem: {_safe_text(body,200)} {_safe_text(question,200)} Answer: {_safe_text(answer,200)}"
                    )
                    processed_data.append(
                        {
                            "text": text,
                            "seq_type": "short",
                            "dataset": "SVAMP",
                            "requires_memory": False,
                            "metadata": {"domain": "math", "type": "word_problem"},
                        }
                    )

            elif dataset_name == "Mini-MBPP":
                for item in dataset:
                    prompt = item.get("prompt") or item.get("task") or item.get("question") or ""
                    code = item.get("code") or item.get("answer") or ""
                    if prompt and code:
                        processed_data.append(
                            {
                                "text": f"Task: {_safe_text(prompt,300)} Code: {_safe_text(code,1000)}",
                                "seq_type": "short",
                                "dataset": "Mini-MBPP",
                                "requires_memory": False,
                                "metadata": {"domain": "code", "language": "python"},
                            }
                        )

            elif dataset_name == "MuSiQue":
                for item in dataset:
                    q = item.get("question") or ""
                    a = item.get("answer") or ""
                    paragraphs = item.get("paragraphs", [])
                    context = " ".join(
                        [_safe_text(p.get("text") if isinstance(p, dict) else p, 200) for p in paragraphs[:3]]
                    )
                    if q and a:
                        processed_data.append(
                            {
                                "text": f"Context: {context} Question: {_safe_text(q,300)} Answer: {_safe_text(a,300)}",
                                "seq_type": "long",
                                "dataset": "MuSiQue",
                                "requires_memory": True,
                                "metadata": {
                                    "hops": item.get("num_hops", 2),
                                    "answerable": item.get("answerable", True),
                                },
                            }
                        )

            elif dataset_name == "NarrativeQA":
                for item in dataset:
                    doc = item.get("document") or {}
                    question = item.get("question") or {}
                    answers = item.get("answers") or []
                    summary = _safe_text(doc.get("summary") if isinstance(doc, dict) else doc, 1000)
                    qtext = question.get("text") if isinstance(question, dict) else question
                    processed_data.append(
                        {
                            "text": f"Story: {summary} Question: {_safe_text(qtext,300)} Answer: {_safe_text(' '.join(answers),400)}",
                            "seq_type": "long",
                            "dataset": "NarrativeQA",
                            "requires_memory": True,
                            "metadata": {
                                "domain": "narrative",
                                "source": (doc.get("kind") if isinstance(doc, dict) else None),
                            },
                        }
                    )

            else:
                # Generic processing for other datasets
                for item in dataset:
                    if isinstance(item, dict):
                        text_fields = ["text", "question", "prompt", "input", "sentence"]
                        text_content = ""
                        for field in text_fields:
                            if field in item and item[field]:
                                text_content = _safe_text(item[field], 500)
                                break
                        if text_content:
                            processed_data.append(
                                {
                                    "text": text_content,
                                    "seq_type": split_type,
                                    "dataset": dataset_name,
                                    "requires_memory": split_type == "long",
                                    "metadata": {"source": "huggingface", "processed": True},
                                }
                            )

            # Save processed dataset
            output_file = dataset_dir / "processed_data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Downloaded {dataset_name}: {len(processed_data)} samples")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}", exc_info=True)
            return False

    def download_all_datasets(self) -> dict[str, bool]:
        """Download all datasets in the curriculum."""
        results = {}

        logger.info("=== DOWNLOADING SHORT/LOCAL DATASETS ===")
        for name, hf_name in self.short_datasets.items():
            results[name] = self.download_dataset(name, hf_name, "short")

        logger.info("=== DOWNLOADING LONG-HORIZON DATASETS ===")
        for name, hf_name in self.long_datasets.items():
            results[name] = self.download_dataset(name, hf_name, "long")

        # Create summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        summary = {
            "total_datasets": total,
            "successful_downloads": successful,
            "failed_downloads": total - successful,
            "download_results": results,
            "output_directory": str(self.output_dir),
            "short_datasets": list(self.short_datasets.keys()),
            "long_datasets": list(self.long_datasets.keys()),
        }

        # Save summary
        with open(self.output_dir / "download_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"=== DOWNLOAD COMPLETE: {successful}/{total} successful ===")
        return results

    def create_mixed_training_data(self, short_ratio: float = 0.45, long_ratio: float = 0.55) -> str:
        """Create mixed training data following curriculum ratios."""

        # Load all processed datasets
        short_data = []
        long_data = []

        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir():
                data_file = dataset_dir / "processed_data.json"
                if data_file.exists():
                    with open(data_file, encoding="utf-8") as f:
                        data = json.load(f)

                    for item in data:
                        if item["seq_type"] == "short":
                            short_data.append(item)
                        else:
                            long_data.append(item)

        logger.info(f"Loaded {len(short_data)} short samples, {len(long_data)} long samples")

        # Calculate mixing ratios
        total_samples = len(short_data) + len(long_data)
        target_short = int(short_ratio * total_samples)
        target_long = int(long_ratio * total_samples)

        # Sample according to ratios
        import random

        random.seed(42)  # Reproducible sampling

        sampled_short = random.sample(short_data, min(target_short, len(short_data)))
        sampled_long = random.sample(long_data, min(target_long, len(long_data)))

        # Combine and shuffle
        mixed_data = sampled_short + sampled_long
        random.shuffle(mixed_data)

        # Save mixed training data
        mixed_file = self.output_dir / "mixed_training_data.json"
        with open(mixed_file, "w", encoding="utf-8") as f:
            json.dump(mixed_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Created mixed training data: {len(sampled_short)} short + {len(sampled_long)} long = {len(mixed_data)} total"
        )
        logger.info(
            f"Ratios: {len(sampled_short)/len(mixed_data):.1%} short, {len(sampled_long)/len(mixed_data):.1%} long"
        )

        return str(mixed_file)


def main():
    """Main dataset download function."""
    logger.info("Starting download of real datasets for Cognate pretraining")

    downloader = CognateDatasetDownloader()
    results = downloader.download_all_datasets()

    # Create mixed training data
    mixed_file = downloader.create_mixed_training_data()

    logger.info("✅ Dataset preparation complete!")
    logger.info(f"Mixed training data saved to: {mixed_file}")

    return results


if __name__ == "__main__":
    results = main()
    successful = sum(1 for success in results.values() if success)
    print(f"SUCCESS: Downloaded {successful}/{len(results)} datasets")
