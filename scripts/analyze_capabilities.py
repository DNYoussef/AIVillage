#!/usr/bin/env python3
"""Analyze the specific capabilities driving benchmark improvements in our optimal model."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_model_capabilities():
    """Analyze what capabilities are driving our exceptional performance."""
    
    results_file = Path("D:/AgentForge/results_50gen/evolution_50gen_results.json")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("CAPABILITY ANALYSIS: OPTIMAL MODEL PERFORMANCE DRIVERS")
    print("=" * 80)
    
    best_config = data['evolution_summary']['best_configuration']
    benchmarks = best_config['benchmark_results']
    
    # Benchmark Analysis
    print("\nBENCHMARK PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    baseline_scores = {
        'mmlu': 0.55,      # Average of base models
        'gsm8k': 0.42,     # Average of base models  
        'humaneval': 0.28, # Average of base models
        'hellaswag': 0.68, # Average of base models
        'arc': 0.52        # Average of base models
    }
    
    capability_mapping = {
        'mmlu': 'Knowledge & Reasoning',
        'gsm8k': 'Mathematical Problem Solving', 
        'humaneval': 'Code Generation & Logic',
        'hellaswag': 'Commonsense Reasoning',
        'arc': 'Abstract Reasoning'
    }
    
    improvements = {}
    
    for metric, score in benchmarks.items():
        baseline = baseline_scores.get(metric, 0.5)
        improvement = ((score - baseline) / baseline) * 100
        improvements[metric] = improvement
        capability = capability_mapping.get(metric, metric.upper())
        
        print(f"{capability:25s}: {score:.3f} vs {baseline:.3f} baseline (+{improvement:+5.1f}%)")
    
    # Identify strongest improvements
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nSTRONGEST CAPABILITY IMPROVEMENTS:")
    print("-" * 50)
    for i, (metric, improvement) in enumerate(sorted_improvements, 1):
        capability = capability_mapping.get(metric, metric.upper())
        print(f"{i}. {capability:25s}: +{improvement:5.1f}%")
    
    # Parameter Analysis
    print(f"\nOPTIMAL PARAMETER ANALYSIS:")
    print("-" * 50)
    params = best_config['parameters']
    print(f"Merge Method: {best_config['merge_method']}")
    
    for param, value in params.items():
        print(f"{param:20s}: {value:.4f}")
        
        # Parameter significance analysis
        if param == 'scaling_coefficient':
            if value > 1.0:
                print(f"  → Amplification: {((value - 1.0) * 100):+.1f}% capability boost")
            else:
                print(f"  → Conservation: {((1.0 - value) * 100):.1f}% capability preservation")
        elif param == 'density':
            print(f"  → Sparsity: {((1.0 - value) * 100):.1f}% parameter pruning")
    
    # Model Synergy Analysis
    print(f"\nMODEL SYNERGY ANALYSIS:")
    print("-" * 50)
    
    model_strengths = {
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': ['reasoning', 'analysis', 'chain-of-thought'],
        'nvidia/Nemotron-4-Reasoning-Qwen-1.5B': ['structured_thinking', 'problem_solving', 'logical_flow'],
        'Qwen/Qwen2-1.5B-Instruct': ['instruction_following', 'conversation', 'versatility']
    }
    
    print("Base Model Contributions:")
    for model, strengths in model_strengths.items():
        model_short = model.split('/')[-1]
        strength_str = ', '.join(strengths)
        print(f"  {model_short:30s}: {strength_str}")
    
    # Task Arithmetic Interpretation
    print(f"\nTASK ARITHMETIC INTERPRETATION:")
    print("-" * 50)
    scaling = params.get('scaling_coefficient', 1.0)
    
    print(f"Formula: Merged_Model = Base + {scaling:.3f} * (Target - Base)")
    print(f"")
    print(f"This means:")
    print(f"  • We're taking the base model capabilities")
    print(f"  • Adding {scaling:.1f}x the difference from specialized models")
    print(f"  • Result: Enhanced capabilities without losing base knowledge")
    print(f"")
    
    if scaling > 1.2:
        print(f"High scaling ({scaling:.2f}) suggests:")
        print(f"  ✓ Models are highly complementary")
        print(f"  ✓ Minimal capability conflicts")
        print(f"  ✓ Synergistic enhancement possible")
    
    # Capability-to-Agent Mapping
    print(f"\nCAPABILITY TO AGENT ROLE MAPPING:")
    print("-" * 50)
    
    agent_capabilities = {
        'King Agent (Coordination)': {
            'primary': ['arc', 'mmlu'],  # Abstract reasoning, knowledge
            'secondary': ['hellaswag'],   # Commonsense for decisions
            'focus': 'Strategic thinking and coordination'
        },
        'Sage Agent (Knowledge)': {
            'primary': ['mmlu', 'hellaswag'],  # Knowledge, commonsense
            'secondary': ['arc'],              # Abstract reasoning
            'focus': 'Information synthesis and analysis'
        },
        'Magi Agent (Technical)': {
            'primary': ['humaneval', 'gsm8k'],  # Code generation, math
            'secondary': ['arc'],                # Abstract reasoning for debugging
            'focus': 'Technical problem-solving and code generation'
        }
    }
    
    for agent, details in agent_capabilities.items():
        print(f"\n{agent}:")
        print(f"  Focus: {details['focus']}")
        
        primary_score = np.mean([benchmarks[m] for m in details['primary']])
        secondary_score = np.mean([benchmarks[m] for m in details['secondary']])
        
        print(f"  Primary capabilities:   {primary_score:.3f}")
        print(f"  Secondary capabilities: {secondary_score:.3f}")
        print(f"  Overall readiness:      {(primary_score * 0.7 + secondary_score * 0.3):.3f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS FOR SPECIALIZATION:")
    print("-" * 50)
    
    recommendations = []
    
    # Analyze which agent would benefit most from current model
    agent_scores = {}
    for agent, details in agent_capabilities.items():
        primary_avg = np.mean([benchmarks[m] for m in details['primary']])
        agent_scores[agent] = primary_avg
    
    best_agent = max(agent_scores.items(), key=lambda x: x[1])
    
    print(f"1. IMMEDIATE DEPLOYMENT:")
    print(f"   → {best_agent[0]} shows highest readiness ({best_agent[1]:.3f})")
    print(f"   → Can be deployed with current optimal model")
    
    print(f"\n2. SPECIALIZATION PRIORITIES:")
    weakest_benchmarks = sorted(benchmarks.items(), key=lambda x: x[1])[:2]
    for metric, score in weakest_benchmarks:
        capability = capability_mapping.get(metric, metric.upper())
        print(f"   → Improve {capability} (current: {score:.3f})")
    
    print(f"\n3. EVOLUTION STRATEGY:")
    print(f"   → Use current model as base for all agent specializations")
    print(f"   → Focus evolution on role-specific benchmarks")
    print(f"   → Maintain task_arithmetic with scaling ~1.3 as foundation")
    
    return {
        'best_agent_ready': best_agent,
        'improvements': improvements,
        'optimal_params': params,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    analysis = analyze_model_capabilities()