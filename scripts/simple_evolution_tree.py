#!/usr/bin/env python3
"""Create a simple ASCII evolutionary tree representation."""

import json
from pathlib import Path

def create_simple_tree():
    """Create simple ASCII evolutionary tree."""
    
    # Load results
    results_file = Path("D:/AgentForge/results/evolution_results.json")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("AGENT FORGE EVOLUTION MERGE - EVOLUTIONARY TREE")
    print("=" * 80)
    
    generations = data['generation_history']
    
    # Method symbols
    method_symbols = {
        'slerp': 'S',
        'linear': 'L', 
        'task_arithmetic': 'T'
    }
    
    print("\nEvolutionary Tree (Method | Fitness)")
    print("-" * 50)
    
    # Track best overall
    best_overall = data['evolution_summary']['best_configuration']
    best_id = best_overall['id']
    
    for gen_idx, gen_data in enumerate(generations):
        population = gen_data['population']
        best_in_gen = max(population, key=lambda x: x['fitness'])
        
        print(f"\nGeneration {gen_idx}:")
        
        for i, individual in enumerate(population):
            method = individual['merge_method']
            fitness = individual['fitness']
            ind_id = individual['id']
            
            symbol = method_symbols.get(method, '?')
            
            # Mark best in generation and best overall
            marker = ""
            if individual == best_in_gen:
                marker += " <-- BEST IN GEN"
            if ind_id == best_id:
                marker += " <-- OVERALL BEST"
                
            fitness_stars = "*" * int(fitness * 10)
            
            print(f"  +-- [{symbol}] Fitness: {fitness:.3f} {fitness_stars}{marker}")
            
            # Show parameters for notable individuals
            if individual == best_in_gen or ind_id == best_id:
                params = individual['parameters']
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"      Parameters: {param_str}")
                
                if 'benchmark_results' in individual:
                    benchmarks = individual['benchmark_results']
                    bench_str = ", ".join([f"{k}={v:.3f}" for k, v in benchmarks.items()])
                    print(f"      Benchmarks: {bench_str}")
    
    print("\n" + "-" * 50)
    print("LEGEND:")
    print("  [S] = SLERP merge")
    print("  [L] = Linear merge") 
    print("  [T] = Task Arithmetic merge")
    print("  * = Fitness level (10 stars = 1.0 fitness)")
    
    print(f"\nBEST CONFIGURATION DISCOVERED:")
    print(f"  Generation: {best_overall['generation']}")
    print(f"  Method: {best_overall['merge_method']}")
    print(f"  Fitness: {best_overall['fitness']:.3f}")
    print(f"  Parameters: {best_overall['parameters']}")
    print(f"  Benchmarks: {best_overall['benchmark_results']}")
    
    # Evolution summary
    print(f"\nEVOLUTION SUMMARY:")
    summary = data['evolution_summary']
    print(f"  Total Generations: {summary['generations_completed']}")
    print(f"  Duration: {summary['duration_seconds']:.1f} seconds")
    print(f"  Best Fitness Achieved: {summary['best_fitness']:.3f}")
    
    # Method evolution
    print(f"\nMETHOD EVOLUTION:")
    method_counts_by_gen = {}
    
    for gen_data in generations:
        gen_num = gen_data['generation']
        method_counts_by_gen[gen_num] = {}
        
        for individual in gen_data['population']:
            method = individual['merge_method']
            method_counts_by_gen[gen_num][method] = method_counts_by_gen[gen_num].get(method, 0) + 1
    
    print("  Gen |  S  |  L  |  T  | Best Method")
    print("  ----|-----|-----|-----|------------")
    
    for gen_num in sorted(method_counts_by_gen.keys()):
        counts = method_counts_by_gen[gen_num]
        s_count = counts.get('slerp', 0)
        l_count = counts.get('linear', 0) 
        t_count = counts.get('task_arithmetic', 0)
        
        # Find best individual in this generation
        gen_data = generations[gen_num]
        best_in_gen = max(gen_data['population'], key=lambda x: x['fitness'])
        best_method = best_in_gen['merge_method']
        
        print(f"   {gen_num:2d} | {s_count:3d} | {l_count:3d} | {t_count:3d} | {best_method}")
    
    print("=" * 80)
    
    # Save to file
    output_file = "D:/AgentForge/results/evolution_tree.txt"
    with open(output_file, 'w') as f:
        # Redirect print to file (simple way)
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        create_simple_tree()
        sys.stdout = original_stdout
    
    print(f"\nEvolutionary tree also saved to: {output_file}")

if __name__ == "__main__":
    create_simple_tree()