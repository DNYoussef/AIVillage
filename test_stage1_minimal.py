#!/usr/bin/env python3
"""Minimal test for Stage1 compression components without external dependencies"""

import json
import os
import sys


# Test basic functionality without torch/bitsandbytes
def test_config_structure():
    """Test configuration structure without dependencies"""
    print("Testing configuration structure...")

    # Test that we can access the config module
    config_path = os.path.join("agent_forge", "compression", "stage1_config.py")
    assert os.path.exists(config_path)

    # Read and check structure
    with open(config_path, encoding="utf-8") as f:
        content = f.read()

    # Check key components exist
    assert "class Stage1Config" in content
    assert "bitnet_learning_rate" in content
    assert "seedlm_block_size" in content
    assert "target_compression_ratio" in content

    print("OK Config structure")


def test_file_structure():
    """Test that all required files exist"""
    print("Testing file structure...")

    required_files = [
        "agent_forge/compression/stage1.py",
        "agent_forge/compression/stage1_bitnet.py",
        "agent_forge/compression/stage1_config.py",
        "agent_forge/compression/seedlm.py",
        "agent_forge/compression/eval_utils.py",
        "eval/hellaswag_sample.jsonl",
        "tests/compression/test_stage1.py",
    ]

    for file_path in required_files:
        assert os.path.exists(file_path), f"Missing file: {file_path}"

    print("OK File structure")


def test_cli_structure():
    """Test CLI structure"""
    print("Testing CLI structure...")

    # Check stage1.py has required functions
    stage1_path = os.path.join("agent_forge", "compression", "stage1.py")
    with open(stage1_path, encoding="utf-8") as f:
        content = f.read()

    # Check key functions and classes exist
    assert "def main()" in content
    assert "def run_stage1_compression" in content
    assert "def load_model_and_tokenizer" in content
    assert "argparse" in content

    print("OK CLI structure")


def test_evaluation_data():
    """Test evaluation data format"""
    print("Testing evaluation data...")

    eval_path = os.path.join("eval", "hellaswag_sample.jsonl")
    with open(eval_path) as f:
        lines = f.readlines()

    assert len(lines) > 0, "No evaluation data found"

    # Check first line structure
    first_line = json.loads(lines[0])
    assert "ctx" in first_line
    assert "endings" in first_line
    assert "label" in first_line
    assert len(first_line["endings"]) >= 2

    print("OK Evaluation data")


def test_implementation_completeness():
    """Test that key implementation components are present"""
    print("Testing implementation completeness...")

    # Check BitNet implementation
    bitnet_path = os.path.join("agent_forge", "compression", "stage1_bitnet.py")
    with open(bitnet_path, encoding="utf-8") as f:
        content = f.read()

    assert "class BitNetLinear" in content
    assert "class RMSNorm" in content
    assert "def convert_to_bitnet" in content
    assert "def quantize_weights" in content

    # Check SeedLM implementation
    seedlm_path = os.path.join("agent_forge", "compression", "seedlm.py")
    with open(seedlm_path, encoding="utf-8") as f:
        content = f.read()

    assert "class SeedLMCompressor" in content
    assert "class LFSRGenerator" in content
    assert "def encode" in content
    assert "def decode" in content

    print("OK Implementation completeness")


def test_success_criteria_constants():
    """Test that success criteria constants are properly defined"""
    print("Testing success criteria constants...")

    config_path = os.path.join("agent_forge", "compression", "stage1_config.py")
    with open(config_path, encoding="utf-8") as f:
        content = f.read()

    # Check constraints
    assert "target_compression_ratio: float = 10.0" in content
    assert "max_accuracy_drop: float = 0.05" in content
    assert "max_memory_gb: float = 16.0" in content

    print("OK Success criteria constants")


def main():
    """Run all minimal tests"""
    print("Running Stage1 Compression Minimal Tests...")
    print("=" * 50)

    try:
        test_file_structure()
        test_config_structure()
        test_cli_structure()
        test_evaluation_data()
        test_implementation_completeness()
        test_success_criteria_constants()

        print("=" * 50)
        print("SUCCESS: All minimal tests passed!")
        print("Stage1 compression pipeline structure is complete")

        # Print next steps
        print("\nNext steps:")
        print("1. Install dependencies: pip install torch transformers bitsandbytes")
        print(
            "2. Run CLI: python -m agent_forge.compression.stage1 --input models/raw/model.pt --output models/compressed/model.stage1.pt"
        )
        print("3. Run tests: python -m pytest tests/compression/test_stage1.py -v")

        # Print success criteria
        print("\nSuccess Criteria:")
        print("- Model compression ratio >= 10x")
        print("- Accuracy drop <= 5%")
        print("- Runs on single GPU with 16GB VRAM")
        print("- Outputs .stage1.pt file")

    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
