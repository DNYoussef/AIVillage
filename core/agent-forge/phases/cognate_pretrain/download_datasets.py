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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CognateDatasetDownloader:
    """Downloads and prepares datasets for Cognate pretraining curriculum."""
    
    def __init__(self, output_dir: str = "./cognate_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.short_datasets = {
            'GSM8K': 'gsm8k',
            'SVAMP': 'ChilleD/SVAMP', 
            'ASDiv': 'EleutherAI/asdiv',
            'Mini-MBPP': 'mbpp',
            'CodeXGLUE': 'microsoft/CodeXGLUE-code-completion-line'
        }
        
        self.long_datasets = {
            'HotpotQA': 'hotpot_qa',
            '2WikiMultiHopQA': '2wikimultihopqa', 
            'MuSiQue': 'musique',
            'QuALITY': 'quality',
            'Qasper': 'qasper',
            'NarrativeQA': 'narrativeqa'
        }
    
    def download_dataset(self, dataset_name: str, hf_name: str, split_type: str = "short") -> bool:
        """Download a single dataset from HuggingFace."""
        try:
            logger.info(f"Downloading {dataset_name} ({hf_name})...")
            
            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name.lower()
            dataset_dir.mkdir(exist_ok=True)
            
            # Download dataset
            dataset = load_dataset(hf_name, split='train', trust_remote_code=True)
            
            # Process based on dataset type
            processed_data = []
            
            if dataset_name == 'GSM8K':
                for item in dataset:
                    processed_data.append({
                        'text': f"Problem: {item['question']} Solution: {item['answer']}",
                        'seq_type': 'short',
                        'dataset': 'GSM8K',
                        'requires_memory': False,
                        'metadata': {'domain': 'math', 'complexity': 'grade_school'}
                    })
            
            elif dataset_name == 'HotpotQA':
                # Use 'fullwiki' version for long-horizon
                dataset = load_dataset(hf_name, 'fullwiki', split='train', trust_remote_code=True)
                for item in dataset:
                    context = ' '.join([f"Document: {ctx}" for ctx in item['context']['sentences'][:5]])  # Limit context
                    processed_data.append({
                        'text': f"Context: {context} Question: {item['question']} Answer: {item['answer']}",
                        'seq_type': 'long',
                        'dataset': 'HotpotQA', 
                        'requires_memory': True,
                        'metadata': {'hops': 2, 'reasoning_type': 'multi_hop'}
                    })
            
            elif dataset_name == 'SVAMP':
                for item in dataset:
                    processed_data.append({
                        'text': f"Problem: {item['Body']} {item['Question']} Answer: {item['Answer']}",
                        'seq_type': 'short',
                        'dataset': 'SVAMP',
                        'requires_memory': False,
                        'metadata': {'domain': 'math', 'type': 'word_problem'}
                    })
            
            elif dataset_name == 'Mini-MBPP':
                for item in dataset:
                    if 'prompt' in item and 'code' in item:
                        processed_data.append({
                            'text': f"Task: {item['prompt']} Code: {item['code']}",
                            'seq_type': 'short', 
                            'dataset': 'Mini-MBPP',
                            'requires_memory': False,
                            'metadata': {'domain': 'code', 'language': 'python'}
                        })
            
            elif dataset_name == 'MuSiQue':
                for item in dataset:
                    if 'question' in item and 'answer' in item:
                        paragraphs = item.get('paragraphs', [])
                        context = ' '.join([p.get('text', '')[:200] for p in paragraphs[:3]])  # Limit context
                        processed_data.append({
                            'text': f"Context: {context} Question: {item['question']} Answer: {item['answer']}",
                            'seq_type': 'long',
                            'dataset': 'MuSiQue',
                            'requires_memory': True,
                            'metadata': {'hops': item.get('num_hops', 2), 'answerable': item.get('answerable', True)}
                        })
            
            elif dataset_name == 'NarrativeQA':
                for item in dataset:
                    if 'document' in item and 'question' in item:
                        summary = item['document']['summary'][:1000]  # Limit length
                        processed_data.append({
                            'text': f"Story: {summary} Question: {item['question']['text']} Answer: {' '.join(item['answers'])}",
                            'seq_type': 'long',
                            'dataset': 'NarrativeQA',
                            'requires_memory': True,
                            'metadata': {'domain': 'narrative', 'source': item['document']['kind']}
                        })
            
            else:
                # Generic processing for other datasets
                for item in dataset:
                    if isinstance(item, dict):
                        # Try to find text-like fields
                        text_fields = ['text', 'question', 'prompt', 'input', 'sentence']
                        text_content = ""
                        for field in text_fields:
                            if field in item and item[field]:
                                text_content = str(item[field])[:500]  # Limit length
                                break
                        
                        if text_content:
                            processed_data.append({
                                'text': text_content,
                                'seq_type': split_type,
                                'dataset': dataset_name,
                                'requires_memory': split_type == 'long',
                                'metadata': {'source': 'huggingface', 'processed': True}
                            })
            
            # Save processed dataset
            output_file = dataset_dir / "processed_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Downloaded {dataset_name}: {len(processed_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}")
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
            'total_datasets': total,
            'successful_downloads': successful,
            'failed_downloads': total - successful,
            'download_results': results,
            'output_directory': str(self.output_dir),
            'short_datasets': list(self.short_datasets.keys()),
            'long_datasets': list(self.long_datasets.keys())
        }
        
        # Save summary
        with open(self.output_dir / "download_summary.json", 'w') as f:
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
                    with open(data_file, encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for item in data:
                        if item['seq_type'] == 'short':
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
        with open(mixed_file, 'w', encoding='utf-8') as f:
            json.dump(mixed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created mixed training data: {len(sampled_short)} short + {len(sampled_long)} long = {len(mixed_data)} total")
        logger.info(f"Ratios: {len(sampled_short)/len(mixed_data):.1%} short, {len(sampled_long)/len(mixed_data):.1%} long")
        
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