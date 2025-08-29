#!/usr/bin/env python3
"""
Comprehensive Playbook Coverage Test
Validates that all 43+ playbooks are covered by the meta-loop orchestration system
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlaybookCoverageValidator:
    """Validates complete playbook coverage in meta-loop system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.playbook_dir = project_root / ".claude" / "playbooks"
        
    def discover_all_playbooks(self) -> Dict[str, List[str]]:
        """Discover all playbooks in the system"""
        playbooks = {
            "meta_conductors": [],
            "loop_conductors": [],
            "specialists": [],
            "loops": [],
            "total_count": 0
        }
        
        # Meta-playbooks (conductors)
        meta_dir = self.playbook_dir / "meta"
        if meta_dir.exists():
            for file in meta_dir.glob("*.playbook.yaml"):
                name = file.stem.replace(".playbook", "")
                playbooks["meta_conductors"].append(name)
        
        # Loop playbooks (root level)
        for file in self.playbook_dir.glob("*-loop.playbook.yaml"):
            name = file.stem.replace(".playbook", "")
            playbooks["loop_conductors"].append(name)
        
        # Specialist playbooks
        specialists_dir = self.playbook_dir / "specialists"
        if specialists_dir.exists():
            for file in specialists_dir.glob("*.yaml"):
                name = file.stem
                playbooks["specialists"].append(name)
            for file in specialists_dir.glob("*.playbook.yaml"):
                name = file.stem.replace(".playbook", "")
                playbooks["specialists"].append(name)
        
        # Loop configurations
        loops_dir = self.playbook_dir / "loops"
        if loops_dir.exists():
            for file in loops_dir.glob("*.yml"):
                if not file.stem.endswith("-template"):
                    name = file.stem
                    playbooks["loops"].append(name)
        
        playbooks["total_count"] = (len(playbooks["meta_conductors"]) + 
                                  len(playbooks["loop_conductors"]) + 
                                  len(playbooks["specialists"]) + 
                                  len(playbooks["loops"]))
        
        return playbooks
    
    def get_expected_43_playbooks(self) -> Set[str]:
        """Define the expected 43 playbooks from the coverage map"""
        expected = {
            # Foundation-Loop coverage
            "p2p", "transport-bench", "fog", "forge", "mobile", "security", 
            "upgrade", "redteam", "obs", "slo", "cve", "drift", "fleet",
            
            # Data-ML-Loop coverage  
            "ingest", "migrate", "rag", "eval", "perf", "compress", "fedlearn", 
            "docs", "docsync",
            
            # Reliability-Loop coverage
            "forensics", "ci", "flakes", "stubs", "consolidate", "refactor", 
            "sev", "upgrade", "docs",
            
            # Delivery-Loop coverage
            "preflight", "mvp", "perf", "obs", "docs", "release", "canary", 
            "chaos", "backtest",
            
            # Economy-Gov-Loop coverage
            "cost", "tokenomics", "dao", "govern", "report",
            
            # Meta conductors
            "selftest", "autopilot",
            
            # Master orchestrator
            "program"
        }
        
        # Remove duplicates (some playbooks appear in multiple loops)
        unique_expected = set(expected)
        logger.info(f"Expected unique playbooks: {len(unique_expected)}")
        return unique_expected
    
    def load_loop_playbook_coverage(self, loop_name: str) -> List[str]:
        """Extract run_playbooks from a loop playbook"""
        loop_file = self.playbook_dir / f"{loop_name}.playbook.yaml"
        if not loop_file.exists():
            return []
        
        try:
            with open(loop_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            covered_playbooks = []
            if 'playbook' in config and 'stages' in config['playbook']:
                for stage in config['playbook']['stages']:
                    if 'run_playbooks' in stage:
                        for playbook_call in stage['run_playbooks']:
                            # Extract playbook name (remove flags and parameters)
                            playbook_name = playbook_call.split()[0].replace('/', '')
                            covered_playbooks.append(playbook_name)
            
            return covered_playbooks
        except Exception as e:
            logger.error(f"Error loading {loop_name}: {e}")
            return []
    
    def validate_coverage_map(self) -> Tuple[bool, Dict]:
        """Validate that the coverage map matches discovered playbooks"""
        discovered = self.discover_all_playbooks()
        expected = self.get_expected_43_playbooks()
        
        # Map discovered playbooks to expected names
        all_discovered = set()
        
        # Add meta conductors
        all_discovered.update(discovered["meta_conductors"])
        
        # Add loop conductors (but convert loop names to base names)
        for loop in discovered["loop_conductors"]:
            base_name = loop.replace("-loop", "")
            all_discovered.add(base_name)
        
        # Add specialists  
        all_discovered.update(discovered["specialists"])
        
        # Add loops
        all_discovered.update(discovered["loops"])
        
        # Check coverage by examining loop playbooks
        loop_coverage = {}
        for loop in ["foundation-loop", "data-ml-loop", "reliability-loop", 
                    "delivery-loop", "economy-gov-loop"]:
            covered = self.load_loop_playbook_coverage(loop)
            loop_coverage[loop] = covered
        
        # Analysis
        covered_by_loops = set()
        for playbooks in loop_coverage.values():
            covered_by_loops.update(playbooks)
        
        missing_from_expected = expected - all_discovered
        extra_discovered = all_discovered - expected
        missing_from_loops = expected - covered_by_loops
        
        validation_result = {
            "total_discovered": discovered["total_count"],
            "expected_count": len(expected),
            "coverage_complete": len(missing_from_expected) == 0,
            "discovered_breakdown": discovered,
            "loop_coverage": loop_coverage,
            "missing_from_expected": list(missing_from_expected),
            "extra_discovered": list(extra_discovered),
            "missing_from_loops": list(missing_from_loops),
            "coverage_by_loops": len(covered_by_loops),
            "all_discovered": list(all_discovered),
            "expected_playbooks": list(expected)
        }
        
        return len(missing_from_expected) == 0, validation_result
    
    def generate_coverage_report(self, validation_result: Dict) -> str:
        """Generate comprehensive coverage report"""
        report_lines = [
            "# Comprehensive Playbook Coverage Report",
            f"**Total Discovered:** {validation_result['total_discovered']}",
            f"**Expected Count:** {validation_result['expected_count']}",
            f"**Coverage Complete:** {'YES' if validation_result['coverage_complete'] else 'NO'}",
            "",
            "## Discovered Playbooks Breakdown",
            f"- **Meta Conductors:** {len(validation_result['discovered_breakdown']['meta_conductors'])} - {validation_result['discovered_breakdown']['meta_conductors']}",
            f"- **Loop Conductors:** {len(validation_result['discovered_breakdown']['loop_conductors'])} - {validation_result['discovered_breakdown']['loop_conductors']}",
            f"- **Specialists:** {len(validation_result['discovered_breakdown']['specialists'])} - {len(validation_result['discovered_breakdown']['specialists'])} total",
            f"- **Loop Configs:** {len(validation_result['discovered_breakdown']['loops'])} - {validation_result['discovered_breakdown']['loops']}",
            "",
            "## Loop Coverage Analysis"
        ]
        
        for loop_name, covered_playbooks in validation_result['loop_coverage'].items():
            report_lines.extend([
                f"### {loop_name}",
                f"**Covers {len(covered_playbooks)} playbooks:** {', '.join(covered_playbooks)}",
                ""
            ])
        
        if validation_result['missing_from_expected']:
            report_lines.extend([
                "## Missing from Expected",
                f"**{len(validation_result['missing_from_expected'])} missing:** {', '.join(validation_result['missing_from_expected'])}",
                ""
            ])
        
        if validation_result['extra_discovered']:
            report_lines.extend([
                "## Extra Discovered (Not in Expected 43)",
                f"**{len(validation_result['extra_discovered'])} extra:** {', '.join(validation_result['extra_discovered'])}",
                ""
            ])
        
        report_lines.extend([
            "## Coverage Summary",
            f"- **Playbooks Covered by Loops:** {validation_result['coverage_by_loops']}",
            f"- **Total Unique Discovered:** {len(validation_result['all_discovered'])}",
            f"- **Coverage Ratio:** {validation_result['coverage_by_loops']/validation_result['expected_count']:.2%}",
            ""
        ])
        
        return "\n".join(report_lines)

def main():
    """Main coverage validation"""
    project_root = Path(__file__).parent.parent
    validator = PlaybookCoverageValidator(project_root)
    
    logger.info("Starting comprehensive playbook coverage validation")
    
    # Validate coverage
    is_complete, results = validator.validate_coverage_map()
    
    # Generate report
    report = validator.generate_coverage_report(results)
    
    # Save results
    results_file = Path(__file__).parent / "comprehensive_coverage_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    report_file = Path(__file__).parent / "COMPREHENSIVE_COVERAGE_REPORT.md"  
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("="*70)
    print(report)
    print("="*70)
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    logger.info(f"Coverage validation {'PASSED' if is_complete else 'NEEDS ATTENTION'}")
    return 0 if is_complete else 1

if __name__ == '__main__':
    exit(main())