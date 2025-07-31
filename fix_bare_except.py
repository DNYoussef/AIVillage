#!/usr/bin/env python3
"""
Fix bare except clauses in Python files.
Replaces bare 'except:' with 'except Exception:'
"""

import re
from pathlib import Path

def fix_bare_except_in_file(file_path: Path) -> bool:
    """Fix bare except clauses in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace bare except clauses
        # Pattern matches:
        # - optional whitespace
        # - 'except'
        # - optional whitespace
        # - ':'
        pattern = r'^(\s*)except\s*:\s*$'
        replacement = r'\1except Exception:'
        
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix bare except clauses in all Python files."""
    project_root = Path(__file__).parent
    
    # Files with bare except clauses (from grep results)
    problematic_files = [
        "benchmarks/hyperag_personalization.py",
        "benchmarks/hyperag_repair_test_suite.py", 
        "agent_forge/prompt_baking.py",
        "agent_forge/evolution/evolution_orchestrator.py",
        "agent_forge/deploy_agent.py",
        "agent_forge/cli.py",
        "digital_twin/monitoring/parent_tracker.py",
        "agent_forge/results_analyzer.py",
        "experimental/training/training/magi_specialization.py",
        "digital_twin/deployment/edge_manager.py",
        "mcp_servers/hyperag/retrieval/importance_flow.py",
        "mcp_servers/hyperag/repair/llm_driver.py",
        "test_dashboard_generator.py",
        "scripts/enhance_compression_mobile.py",
        "scripts/download_benchmarks.py",
        "tests/conftest.py",
        "scripts/run_full_agent_forge.py",
        "scripts/run_agent_forge.py",
        "migration/knowledge_graph_bootstrapper.py",
        "tests/test_adas_secure_standalone.py",
        "production/evolution/evolution/math_fitness.py",
        "tests/test_pipeline_integration.py",
        "production/evolution/evolution/deploy_winner.py",
        "tests/mcp_servers/test_hyperag_server.py",
        "tests/production/test_rag_system.py"
    ]
    
    files_fixed = 0
    files_processed = 0
    
    for file_path_str in problematic_files:
        file_path = project_root / file_path_str
        if file_path.exists():
            files_processed += 1
            if fix_bare_except_in_file(file_path):
                files_fixed += 1
                print(f"✓ Fixed bare except clauses in {file_path_str}")
            else:
                print(f"- No changes needed in {file_path_str}")
        else:
            print(f"⚠ File not found: {file_path_str}")
    
    print(f"\nSummary:")
    print(f"Files processed: {files_processed}")
    print(f"Files fixed: {files_fixed}")
    
    if files_fixed > 0:
        print(f"\n✓ Fixed bare except clauses in {files_fixed} files")
        print("✓ This improves error handling and code quality")
    else:
        print("\n- No bare except clauses found to fix")
    
    return 0

if __name__ == "__main__":
    exit(main())