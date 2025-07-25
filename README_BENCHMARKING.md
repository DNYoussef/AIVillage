# 🏆 Agent Forge Comprehensive Benchmarking System

## Overview

The Agent Forge benchmarking system provides comprehensive evaluation of trained models against standardized benchmarks and comparison with baseline 1.5B and frontier models. The system generates detailed W&B reports and publication-ready performance summaries.

## 🎯 Benchmarks Included

### Core Benchmarks
- **MMLU** (Massive Multitask Language Understanding) - 57 subjects across knowledge domains
- **GSM8K** (Grade School Math 8K) - Mathematical reasoning problems
- **HumanEval** - Python code generation tasks
- **HellaSwag** - Commonsense reasoning
- **ARC** (AI2 Reasoning Challenge) - Scientific reasoning

### Comparison Models
- **Baseline Models** (1.5B range): DialoGPT-large (762M), OPT-1.3B, GPT-Neo-1.3B
- **Frontier Models**: Phi-2 (2.7B), Mistral-7B-Instruct

## 🚀 Quick Start

### 1. Full Agent Forge Pipeline Evaluation
```bash
# Evaluate all pipeline outputs with comprehensive reporting
python run_agent_forge_benchmark.py --full

# Quick evaluation for testing (limited samples)
python run_agent_forge_benchmark.py --quick
```

### 2. Individual Model Evaluation
```bash
# Benchmark specific model
python agent_forge/benchmark_runner.py \
  --model-path ./mastery_output/final_model \
  --model-name agent-forge-mastery \
  --output-dir ./benchmark_results
```

### 3. Model Validation
```bash
# Check which models are available for benchmarking
python run_agent_forge_benchmark.py --validate-only
```

## 📊 Output Reports

### 1. Executive Summary
- **File**: `benchmark_results/executive_summary.md`
- **Content**: High-level performance summary and model rankings

### 2. Detailed Comparison Report
- **File**: `benchmark_results/agent_forge_detailed_comparison.md`
- **Content**: Comprehensive model-by-model analysis with insights

### 3. Cross-Stage Analysis
- **File**: `benchmark_results/cross_stage_report.md`
- **Content**: Performance evolution across pipeline stages

### 4. W&B Dashboard
- **Project**: `agent-forge-comprehensive-benchmark`
- **Features**: Interactive visualizations, statistical analysis, model comparisons

## 🏗️ System Architecture

```
Agent Forge Benchmarking System
├── benchmark_suite.py          # Core evaluation framework
├── benchmark_runner.py         # Automated benchmark orchestration
├── run_agent_forge_benchmark.py # Main integration script
└── Individual Evaluators:
    ├── MMLUEvaluator           # Knowledge & reasoning
    ├── GSM8KEvaluator          # Mathematical reasoning
    ├── HumanEvalEvaluator      # Code generation
    └── ComprehensiveBenchmark  # Overall orchestration
```

## 📈 Performance Analysis Features

### Statistical Analysis
- **Percentile Rankings**: Compare against baseline and frontier models
- **T-Tests**: Statistical significance testing
- **Confidence Intervals**: Performance variance analysis
- **Effect Sizes**: Practical significance measurement

### Visualization
- **Performance Radar Charts**: Multi-benchmark comparison
- **Bar Charts**: Model-by-model benchmark scores
- **Trend Analysis**: Cross-stage performance evolution
- **Heatmaps**: Benchmark correlation analysis

## 🎛️ Configuration Options

### Basic Configuration
```python
config = BenchmarkConfig(
    model_path="./your_model",
    model_name="your-model-name",
    run_mmlu=True,
    run_gsm8k=True,
    run_humaneval=True,
    batch_size=4,
    max_samples=None,  # Full evaluation
    device="cuda",
    precision="fp16"
)
```

### Advanced Options
```bash
# Skip specific comparisons for faster evaluation
python run_agent_forge_benchmark.py --skip-baselines --skip-frontier

# Custom results directory
python run_agent_forge_benchmark.py --results-dir ./custom_results

# Quick mode with limited samples (for testing)
python run_agent_forge_benchmark.py --quick
```

## 📋 Expected Results Format

### Model Performance Summary
```json
{
  "model_name": "agent-forge-mastery",
  "average_score": 0.742,
  "benchmark_scores": {
    "MMLU": 0.678,
    "GSM8K": 0.823,
    "HumanEval": 0.456,
    "HellaSwag": 0.891,
    "ARC": 0.654
  },
  "wins_vs_baseline": 4.2,
  "wins_vs_frontier": 1.8
}
```

### Statistical Analysis
```json
{
  "MMLU": {
    "target_score": 0.678,
    "baseline_mean": 0.543,
    "baseline_percentile": 87.3,
    "frontier_percentile": 34.2,
    "statistical_significance": true
  }
}
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Not Found
```bash
# Check model availability
python run_agent_forge_benchmark.py --validate-only

# Expected output:
# mastery_trained        : ✅ Valid
# evomerge_best         : ❌ Missing
```

#### 2. CUDA Out of Memory
```bash
# Use smaller batch size
python agent_forge/benchmark_suite.py --batch-size 2

# Or use CPU evaluation
python agent_forge/benchmark_suite.py --device cpu
```

#### 3. Dataset Download Issues
```bash
# Pre-download datasets
python -c "
from datasets import load_dataset
load_dataset('cais/mmlu', 'abstract_algebra')
load_dataset('gsm8k', 'main')
load_dataset('openai_humaneval')
"
```

### Performance Optimization

#### For RTX 2060 SUPER (8GB VRAM)
```python
# Optimized configuration
config = BenchmarkConfig(
    batch_size=2,           # Reduced batch size
    precision="fp16",       # Half precision
    max_samples=1000,       # Limit samples if needed
    device="cuda"
)
```

#### Quick Testing Mode
```bash
# Fast evaluation with reduced samples
python run_agent_forge_benchmark.py --quick --skip-frontier
```

## 📊 Benchmark Interpretation Guide

### Score Ranges
- **0.9-1.0**: Excellent (State-of-the-art performance)
- **0.8-0.9**: Very Good (Strong performance)
- **0.7-0.8**: Good (Above average)
- **0.6-0.7**: Fair (Average performance)
- **0.5-0.6**: Below Average (Needs improvement)
- **<0.5**: Poor (Significant improvement needed)

### Benchmark-Specific Expectations

#### MMLU (Knowledge & Reasoning)
- **Random Baseline**: 0.25 (4-choice multiple choice)
- **Strong Performance**: >0.7
- **Human Performance**: ~0.9

#### GSM8K (Math Reasoning)
- **Random Baseline**: ~0.0
- **Strong Performance**: >0.6
- **Human Performance**: ~0.9

#### HumanEval (Code Generation)
- **Random Baseline**: ~0.0
- **Strong Performance**: >0.3
- **Human Performance**: ~0.8

## 🔗 Integration with W&B

### Automatic Logging
The system automatically logs to W&B:
- Individual benchmark scores
- Model comparisons
- Statistical analysis
- Performance visualizations

### Manual W&B Setup
```python
import wandb

wandb.init(
    project="agent-forge-benchmark",
    name="custom-evaluation",
    config={
        "model": "your-model",
        "benchmarks": ["MMLU", "GSM8K", "HumanEval"]
    }
)
```

### Viewing Results
```bash
# View in browser
wandb.ai/agent-forge/agent-forge-comprehensive-benchmark

# Download results programmatically
wandb.Api().runs("agent-forge/agent-forge-comprehensive-benchmark")
```

## 📚 Expected Outputs

After running the benchmark, you'll find:

```
benchmark_results/
├── executive_summary.md              # High-level results
├── agent_forge_detailed_comparison.md # Detailed analysis
├── cross_stage_comparison.json       # Pipeline evolution
├── agent_forge_model_comparison.json # Raw comparison data
└── individual_models/
    ├── agent-forge-mastery/
    │   ├── MMLU_results.json
    │   ├── GSM8K_results.json
    │   ├── HumanEval_results.json
    │   └── performance_report.md
    └── agent-forge-evomerge/
        └── [similar structure]
```

## 🎯 Publication-Ready Results

The system generates publication-ready outputs:

1. **Performance Tables** (LaTeX/Markdown format)
2. **Statistical Significance Tests** (p-values, effect sizes)
3. **Confidence Intervals** (95% CI for all metrics)
4. **Comparison Charts** (PNG/SVG for papers)
5. **Executive Summaries** (For stakeholder presentations)

## 💡 Best Practices

### For Research Papers
1. Run full evaluation (not quick mode)
2. Include baseline and frontier comparisons
3. Report confidence intervals
4. Use statistical significance testing
5. Include detailed methodology section

### For Development
1. Use quick mode for iterative testing
2. Focus on specific benchmarks during development
3. Compare against previous versions
4. Monitor performance trends over time

### For Production
1. Evaluate on held-out test sets
2. Include safety and bias evaluations
3. Test on domain-specific benchmarks
4. Validate performance stability

---

The Agent Forge benchmarking system provides comprehensive, automated evaluation with publication-ready results and detailed statistical analysis. Use it to validate model performance, compare approaches, and generate evidence for research publications.
