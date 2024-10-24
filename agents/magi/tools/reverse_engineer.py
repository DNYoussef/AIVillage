"""Reverse engineering functionality for MAGI."""

from typing import Dict, Any, List, Optional
import asyncio
import logging
import math
import json
from datetime import datetime
import angr
import r2pipe
import yara
import networkx as nx
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ReverseEngineer:
    """
    Handles reverse engineering of programs.
    
    Capabilities:
    - Binary analysis
    - Symbolic execution
    - String analysis
    - Crypto detection
    - Call graph analysis
    - Obfuscation detection
    - Data flow analysis
    - Design pattern recognition
    - API usage analysis
    - Hypothesis generation and testing
    - Self-reflection and learning
    """
    
    def __init__(self, llm=None, continuous_learner=None):
        """
        Initialize reverse engineer.
        
        Args:
            llm: Language model for analysis
            continuous_learner: Continuous learning component
        """
        self.llm = llm
        self.continuous_learner = continuous_learner
        self.project = None
        self.cfg = None
        self.call_graph = None
        self.analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_program(self, program_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a program.
        
        Args:
            program_path: Path to program binary
            
        Returns:
            Analysis results
        """
        # Generate initial hypotheses
        hypotheses = await self._generate_hypotheses(program_path)
        
        # Initialize analysis
        self.project = angr.Project(program_path, auto_load_libs=False)
        self.cfg = self.project.analyses.CFGFast()
        self.call_graph = self.cfg.graph
        
        # Run analysis tasks in parallel
        tasks = [
            self.analyze_main(),
            self.perform_symbolic_execution(),
            self.extract_and_analyze_strings(),
            self.identify_crypto_usage(),
            self.analyze_call_graph(),
            self.detect_obfuscation(),
            self.perform_data_flow_analysis(),
            self.identify_design_patterns(),
            self.analyze_api_usage()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        main_analysis, sym_execution, strings_analysis, crypto_analysis, \
        call_graph_analysis, obfuscation_analysis, data_flow_analysis, \
        design_patterns, api_usage = results
        
        # Test hypotheses
        hypothesis_results = await self._test_hypotheses(
            hypotheses,
            main_analysis,
            sym_execution,
            strings_analysis,
            crypto_analysis,
            call_graph_analysis,
            obfuscation_analysis,
            data_flow_analysis,
            design_patterns,
            api_usage
        )
        
        # Generate comprehensive report
        report = await self.generate_comprehensive_report(
            main_analysis, sym_execution, strings_analysis,
            crypto_analysis, call_graph_analysis, obfuscation_analysis,
            data_flow_analysis, design_patterns, api_usage,
            hypothesis_results
        )
        
        # Perform self-reflection
        reflection = await self._reflect_on_analysis(
            program_path,
            hypotheses,
            hypothesis_results,
            report
        )
        
        # Generate replication code
        replicated_functions = await self.replicate_key_functions(report)
        
        # Update continuous learning
        if self.continuous_learner:
            await self.continuous_learner.learn_from_analysis(
                program_path,
                report,
                reflection
            )
        
        # Record analysis
        self.analysis_history.append({
            'program_path': program_path,
            'hypotheses': hypotheses,
            'hypothesis_results': hypothesis_results,
            'report': report,
            'reflection': reflection,
            'timestamp': datetime.now()
        })
        
        return {
            'report': report,
            'replicated_functions': replicated_functions,
            'hypotheses': hypotheses,
            'hypothesis_results': hypothesis_results,
            'reflection': reflection
        }
    
    async def _generate_hypotheses(self, program_path: str) -> List[Dict[str, Any]]:
        """Generate hypotheses about program behavior."""
        if not self.llm:
            return []
        
        prompt = f"""
        Generate hypotheses about the behavior of this program: {program_path}
        
        Consider:
        1. Potential functionality and purpose
        2. Expected components and structure
        3. Likely algorithms and data structures
        4. Security mechanisms
        5. Performance characteristics
        
        For each hypothesis, provide:
        1. The hypothesis statement
        2. Expected evidence
        3. Test approach
        4. Success criteria
        
        Format as JSON list of dictionaries.
        """
        
        response = await self.llm.complete(prompt)
        return json.loads(response.text)
    
    async def _test_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        *analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Test hypotheses against analysis results."""
        results = []
        
        for hypothesis in hypotheses:
            evidence = []
            for result in analysis_results:
                relevant_evidence = self._find_relevant_evidence(
                    hypothesis,
                    result
                )
                evidence.extend(relevant_evidence)
            
            success = self._evaluate_hypothesis(
                hypothesis,
                evidence,
                hypothesis['success_criteria']
            )
            
            results.append({
                'hypothesis': hypothesis['statement'],
                'evidence': evidence,
                'success': success,
                'confidence': self._calculate_confidence(evidence)
            })
        
        return results
    
    def _find_relevant_evidence(
        self,
        hypothesis: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find evidence relevant to a hypothesis."""
        evidence = []
        
        # Search for evidence in analysis results
        # This is a placeholder - implement actual evidence finding logic
        
        return evidence
    
    def _evaluate_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        criteria: Dict[str, Any]
    ) -> bool:
        """Evaluate if evidence supports a hypothesis."""
        # Implement hypothesis evaluation
        # This is a placeholder - implement actual evaluation logic
        return True
    
    def _calculate_confidence(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate confidence in evidence."""
        # Implement confidence calculation
        # This is a placeholder - implement actual calculation
        return 0.8
    
    async def _reflect_on_analysis(
        self,
        program_path: str,
        hypotheses: List[Dict[str, Any]],
        hypothesis_results: List[Dict[str, Any]],
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform self-reflection on analysis process."""
        if not self.llm:
            return {}
        
        prompt = f"""
        Reflect on the analysis of this program:
        
        Program: {program_path}
        Hypotheses: {json.dumps(hypotheses, indent=2)}
        Results: {json.dumps(hypothesis_results, indent=2)}
        Report: {json.dumps(report, indent=2)}
        
        Provide:
        1. Effectiveness of initial hypotheses
        2. Quality of evidence gathering
        3. Confidence in conclusions
        4. Areas for improvement
        5. Lessons learned
        6. Recommendations for future analysis
        
        Format as JSON.
        """
        
        response = await self.llm.complete(prompt)
        return json.loads(response.text)
    
    async def get_analysis_insights(self) -> Dict[str, Any]:
        """Get insights from analysis history."""
        if not self.llm or not self.analysis_history:
            return {}
        
        prompt = f"""
        Analyze this reverse engineering history:
        {json.dumps(self.analysis_history, indent=2)}
        
        Provide insights on:
        1. Common patterns across programs
        2. Successful analysis techniques
        3. Areas for improvement
        4. Recommendations for future analysis
        
        Format as JSON.
        """
        
        response = await self.llm.complete(prompt)
        return json.loads(response.text)
    
    async def analyze_main(self) -> Dict[str, Any]:
        """Analyze program entry point and main function."""
        main_func = self.cfg.functions.get('main')
        if not main_func:
            return {"error": "Main function not found"}
        
        analysis = {
            'address': hex(main_func.addr),
            'size': main_func.size,
            'blocks': len(main_func.blocks),
            'calls': [hex(call.addr) for call in main_func.get_call_sites()],
            'local_vars': len(main_func.local_variables),
            'arguments': len(main_func.arguments)
        }
        
        return analysis
    
    async def perform_symbolic_execution(self) -> Dict[str, Any]:
        """Perform symbolic execution to find paths and constraints."""
        simgr = self.project.factory.simulation_manager()
        simgr.explore(find=lambda s: "success" in s.posix.dumps(1))
        
        paths = []
        for state in simgr.found:
            path = {
                'constraints': [str(c) for c in state.solver.constraints],
                'input_values': state.posix.dumps(0),
                'output': state.posix.dumps(1)
            }
            paths.append(path)
        
        return {'paths': paths}
    
    async def extract_and_analyze_strings(self) -> Dict[str, Any]:
        """Extract and analyze strings from binary."""
        r2 = r2pipe.open(self.project.filename)
        strings = r2.cmdj('izj')  # Get strings in JSON format
        
        # Categorize strings
        categories = {
            'urls': [],
            'paths': [],
            'commands': [],
            'api_keys': [],
            'other': []
        }
        
        for s in strings:
            string = s['string']
            if string.startswith(('http://', 'https://')):
                categories['urls'].append(string)
            elif '/' in string or '\\' in string:
                categories['paths'].append(string)
            elif any(cmd in string.lower() for cmd in ['exec', 'system', 'cmd']):
                categories['commands'].append(string)
            elif len(string) > 20 and any(c in string for c in ['_', '-', '.']):
                categories['api_keys'].append(string)
            else:
                categories['other'].append(string)
        
        return categories
    
    async def identify_crypto_usage(self) -> Dict[str, Any]:
        """Identify cryptographic operations and algorithms."""
        # Common crypto constants
        crypto_constants = {
            'AES_SBOX': bytes.fromhex('637c777bf26b6fc53001672bfed7ab76'),
            'SHA256_IV': bytes.fromhex('6a09e667bb67ae853c6ef372a54ff53a'),
            'DES_SBOX': bytes.fromhex('3432231100776655')
        }
        
        findings = {
            'constants': [],
            'functions': [],
            'algorithms': []
        }
        
        # Search for crypto constants
        for name, constant in crypto_constants.items():
            refs = self.project.loader.memory.find_all(constant)
            if refs:
                findings['constants'].append({
                    'name': name,
                    'locations': [hex(ref) for ref in refs]
                })
        
        # Look for crypto function names
        crypto_funcs = ['aes', 'des', 'sha', 'md5', 'rsa', 'encrypt', 'decrypt']
        for func in self.cfg.functions.values():
            for pattern in crypto_funcs:
                if pattern in func.name.lower():
                    findings['functions'].append({
                        'name': func.name,
                        'address': hex(func.addr)
                    })
        
        return findings
    
    async def analyze_call_graph(self) -> Dict[str, Any]:
        """Analyze program call graph structure."""
        analysis = {
            'total_functions': len(self.cfg.functions),
            'entry_points': [],
            'leaf_functions': [],
            'highly_connected': [],
            'cycles': []
        }
        
        # Find entry points (functions with no callers)
        for func in self.cfg.functions.values():
            if not func.predecessors:
                analysis['entry_points'].append({
                    'name': func.name,
                    'address': hex(func.addr)
                })
        
        # Find leaf functions (functions that don't call others)
        for func in self.cfg.functions.values():
            if not func.successors:
                analysis['leaf_functions'].append({
                    'name': func.name,
                    'address': hex(func.addr)
                })
        
        # Find highly connected functions
        for func in self.cfg.functions.values():
            if len(func.predecessors) + len(func.successors) > 10:
                analysis['highly_connected'].append({
                    'name': func.name,
                    'address': hex(func.addr),
                    'callers': len(func.predecessors),
                    'callees': len(func.successors)
                })
        
        # Find cycles in call graph
        cycles = nx.simple_cycles(self.call_graph)
        for cycle in cycles:
            analysis['cycles'].append([
                self.cfg.functions[addr].name for addr in cycle
            ])
        
        return analysis
    
    async def detect_obfuscation(self) -> Dict[str, Any]:
        """Detect code obfuscation techniques."""
        analysis = {
            'techniques': [],
            'suspicious_patterns': [],
            'entropy': {}
        }
        
        # Check for common obfuscation techniques
        techniques = {
            'string_encryption': self._check_string_encryption(),
            'control_flow_flattening': self._check_control_flow_flattening(),
            'instruction_substitution': self._check_instruction_substitution(),
            'dynamic_code': self._check_dynamic_code()
        }
        
        for technique, detected in techniques.items():
            if detected:
                analysis['techniques'].append(technique)
        
        # Calculate entropy for different sections
        for section in self.project.loader.main_object.sections:
            if section.is_executable:
                entropy = self._calculate_entropy(section.content)
                analysis['entropy'][section.name] = entropy
        
        return analysis
    
    def _check_string_encryption(self) -> bool:
        """Check for string encryption."""
        # Look for patterns indicating string encryption
        return False  # Placeholder
    
    def _check_control_flow_flattening(self) -> bool:
        """Check for control flow flattening."""
        # Look for patterns indicating flattened control flow
        return False  # Placeholder
    
    def _check_instruction_substitution(self) -> bool:
        """Check for instruction substitution."""
        # Look for unusual instruction patterns
        return False  # Placeholder
    
    def _check_dynamic_code(self) -> bool:
        """Check for dynamic code generation."""
        # Look for patterns indicating dynamic code
        return False  # Placeholder
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        entropy = 0
        for x in range(256):
            p_x = data.count(x) / len(data)
            if p_x > 0:
                entropy += -p_x * math.log2(p_x)
        
        return entropy
    
    async def perform_data_flow_analysis(self) -> Dict[str, Any]:
        """Analyze data flow through the program."""
        analysis = {
            'sources': [],
            'sinks': [],
            'flows': []
        }
        
        # Identify data sources (e.g., input functions)
        sources = ['scanf', 'fgets', 'read']
        for func in self.cfg.functions.values():
            for source in sources:
                if source in func.name:
                    analysis['sources'].append({
                        'name': func.name,
                        'address': hex(func.addr)
                    })
        
        # Identify data sinks (e.g., output functions)
        sinks = ['printf', 'fprintf', 'write']
        for func in self.cfg.functions.values():
            for sink in sinks:
                if sink in func.name:
                    analysis['sinks'].append({
                        'name': func.name,
                        'address': hex(func.addr)
                    })
        
        # Track data flows between sources and sinks
        for source in analysis['sources']:
            for sink in analysis['sinks']:
                paths = nx.all_simple_paths(
                    self.call_graph,
                    source['address'],
                    sink['address']
                )
                for path in paths:
                    analysis['flows'].append({
                        'source': source['name'],
                        'sink': sink['name'],
                        'path': [hex(addr) for addr in path]
                    })
        
        return analysis
    
    async def identify_design_patterns(self) -> Dict[str, Any]:
        """Identify common design patterns in the code."""
        patterns = {
            'singleton': self._find_singleton_pattern(),
            'factory': self._find_factory_pattern(),
            'observer': self._find_observer_pattern(),
            'decorator': self._find_decorator_pattern()
        }
        
        return {
            name: instances for name, instances in patterns.items()
            if instances
        }
    
    def _find_singleton_pattern(self) -> List[Dict[str, Any]]:
        """Find singleton pattern implementations."""
        instances = []
        
        # Look for classes with private constructors and static instance
        for func in self.cfg.functions.values():
            if (
                func.name.startswith('_')  # Private constructor
                and any(  # Static instance access
                    'get_instance' in f.name
                    for f in func.successors
                )
            ):
                instances.append({
                    'class': func.name,
                    'constructor': hex(func.addr),
                    'instance_getter': [
                        hex(f.addr) for f in func.successors
                        if 'get_instance' in f.name
                    ]
                })
        
        return instances
    
    def _find_factory_pattern(self) -> List[Dict[str, Any]]:
        """Find factory pattern implementations."""
        instances = []
        
        # Look for create/make methods returning different types
        factory_methods = [
            func for func in self.cfg.functions.values()
            if any(name in func.name.lower() for name in ['create', 'make', 'build'])
            and len(func.successors) > 2  # Multiple creation paths
        ]
        
        for func in factory_methods:
            instances.append({
                'factory': func.name,
                'address': hex(func.addr),
                'products': [
                    hex(f.addr) for f in func.successors
                ]
            })
        
        return instances
    
    def _find_observer_pattern(self) -> List[Dict[str, Any]]:
        """Find observer pattern implementations."""
        instances = []
        
        # Look for notify/update methods with multiple observers
        observer_methods = [
            func for func in self.cfg.functions.values()
            if any(name in func.name.lower() for name in ['notify', 'update'])
            and len(func.predecessors) > 2  # Multiple observers
        ]
        
        for func in observer_methods:
            instances.append({
                'subject': func.name,
                'address': hex(func.addr),
                'observers': [
                    hex(f.addr) for f in func.predecessors
                ]
            })
        
        return instances
    
    def _find_decorator_pattern(self) -> List[Dict[str, Any]]:
        """Find decorator pattern implementations."""
        instances = []
        
        # Look for wrapper classes with same interface
        for func in self.cfg.functions.values():
            if (
                'wrapper' in func.name.lower()
                and len(func.successors) == 1  # Wraps one component
                and func.signature == func.successors[0].signature  # Same interface
            ):
                instances.append({
                    'decorator': func.name,
                    'address': hex(func.addr),
                    'component': hex(func.successors[0].addr)
                })
        
        return instances
    
    async def analyze_api_usage(self) -> Dict[str, Any]:
        """Analyze external API usage."""
        analysis = {
            'imports': [],
            'exports': [],
            'libraries': [],
            'syscalls': []
        }
        
        # Analyze imports
        for imp in self.project.loader.main_object.imports.values():
            analysis['imports'].append({
                'name': imp.name,
                'address': hex(imp.addr),
                'type': imp.type
            })
        
        # Analyze exports
        for exp in self.project.loader.main_object.exports.values():
            analysis['exports'].append({
                'name': exp.name,
                'address': hex(exp.addr),
                'type': exp.type
            })
        
        # Analyze library dependencies
        for lib in self.project.loader.requested_names:
            analysis['libraries'].append(lib)
        
        # Analyze system calls
        for func in self.cfg.functions.values():
            for block in func.blocks:
                for insn in block.capstone.insns:
                    if insn.mnemonic == 'syscall':
                        analysis['syscalls'].append({
                            'address': hex(insn.address),
                            'function': func.name
                        })
        
        return analysis
    
    async def generate_comprehensive_report(
        self,
        main_analysis: Dict[str, Any],
        sym_execution: Dict[str, Any],
        strings_analysis: Dict[str, Any],
        crypto_analysis: Dict[str, Any],
        call_graph_analysis: Dict[str, Any],
        obfuscation_analysis: Dict[str, Any],
        data_flow_analysis: Dict[str, Any],
        design_patterns: Dict[str, Any],
        api_usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        return {
            'program_structure': {
                'main_function': main_analysis,
                'call_graph': call_graph_analysis,
                'design_patterns': design_patterns
            },
            'behavior_analysis': {
                'symbolic_execution': sym_execution,
                'data_flow': data_flow_analysis,
                'api_usage': api_usage
            },
            'security_analysis': {
                'strings': strings_analysis,
                'crypto': crypto_analysis,
                'obfuscation': obfuscation_analysis
            },
            'recommendations': await self._generate_recommendations(
                main_analysis, sym_execution, strings_analysis,
                crypto_analysis, call_graph_analysis, obfuscation_analysis,
                data_flow_analysis, design_patterns, api_usage
            )
        }
    
    async def _generate_recommendations(self, *analyses) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Add recommendations based on each analysis
        # This is a placeholder - implement actual recommendation logic
        
        return recommendations
    
    async def replicate_key_functions(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Generate Python code to replicate key functions."""
        replicated = {}
        
        # Identify and replicate key functions
        # This is a placeholder - implement actual replication logic
        
        return replicated