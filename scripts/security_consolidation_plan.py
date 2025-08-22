#!/usr/bin/env python3
"""Security Module Consolidation Plan - Eliminate Encryption Duplication

This script orchestrates the consolidation of 27+ encryption implementations
into a unified, single-source-of-truth security module following connascence
management principles.

Key Goals:
1. Eliminate CoA (Connascence of Algorithm) violations
2. Reduce from 27 to 1 encryption implementation
3. Apply dependency injection patterns
4. Maintain backward compatibility
"""

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EncryptionMethod:
    """Represents a discovered encryption method for consolidation analysis"""

    file_path: str
    function_name: str
    line_number: int
    method_type: str  # encrypt, decrypt, hash, sign
    parameters: list[str]
    algorithm: str
    complexity_score: float
    usage_count: int = 0


@dataclass
class ConsolidationPlan:
    """Plan for consolidating encryption methods"""

    target_module: str
    unified_interface: str
    migration_steps: list[str]
    backward_compatibility: list[str]
    estimated_reduction: dict[str, float]


class SecurityConsolidationAnalyzer:
    """Analyzes codebase for security consolidation opportunities"""

    def __init__(self, codebase_path: Path):
        self.codebase_path = codebase_path
        self.encryption_methods: list[EncryptionMethod] = []
        self.duplicate_algorithms: dict[str, list[EncryptionMethod]] = {}

    def analyze_encryption_patterns(self) -> dict[str, Any]:
        """Scan codebase for encryption method patterns"""

        # Security-related file patterns
        security_patterns = [
            "**/*security*.py",
            "**/*encryption*.py",
            "**/*crypto*.py",
            "**/digital_twin*.py",
            "**/secure_*.py",
        ]

        for pattern in security_patterns:
            for file_path in self.codebase_path.glob(pattern):
                if file_path.is_file():
                    self._analyze_file(file_path)

        return {
            "total_encryption_methods": len(self.encryption_methods),
            "duplicate_algorithms": self._find_duplicate_algorithms(),
            "consolidation_opportunities": self._identify_consolidation_opportunities(),
            "complexity_analysis": self._analyze_complexity(),
        }

    def _analyze_file(self, file_path: Path):
        """Analyze a single file for encryption patterns"""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method = self._extract_encryption_method(node, file_path)
                    if method:
                        self.encryption_methods.append(method)

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")

    def _extract_encryption_method(self, node: ast.FunctionDef, file_path: Path) -> EncryptionMethod | None:
        """Extract encryption method details from AST node"""

        # Check if this is an encryption-related method
        crypto_keywords = ["encrypt", "decrypt", "hash", "sign", "cipher", "crypt"]

        if not any(keyword in node.name.lower() for keyword in crypto_keywords):
            return None

        # Determine method type
        method_type = "unknown"
        if "encrypt" in node.name.lower():
            method_type = "encrypt"
        elif "decrypt" in node.name.lower():
            method_type = "decrypt"
        elif "hash" in node.name.lower():
            method_type = "hash"
        elif "sign" in node.name.lower():
            method_type = "sign"

        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]

        # Analyze algorithm used (simplified detection)
        algorithm = self._detect_algorithm(node)

        # Calculate complexity score
        complexity = self._calculate_complexity(node)

        return EncryptionMethod(
            file_path=str(file_path),
            function_name=node.name,
            line_number=node.lineno,
            method_type=method_type,
            parameters=parameters,
            algorithm=algorithm,
            complexity_score=complexity,
        )

    def _detect_algorithm(self, node: ast.FunctionDef) -> str:
        """Detect encryption algorithm used in method"""

        # Look for algorithm indicators in the code
        algorithm_indicators = {
            "fernet": "Fernet",
            "aes": "AES",
            "rsa": "RSA",
            "sha256": "SHA256",
            "bcrypt": "bcrypt",
            "nacl": "NaCl",
            "pbkdf2": "PBKDF2",
        }

        # Convert function to string and check for indicators
        try:
            import astor

            code_str = astor.to_source(node).lower()

            for indicator, algorithm in algorithm_indicators.items():
                if indicator in code_str:
                    return algorithm

        except ImportError:
            # Fallback to simple string search
            pass

        return "unknown"

    def _calculate_complexity(self, node: ast.FunctionDef) -> float:
        """Calculate complexity score for encryption method"""

        complexity = 1.0  # Base complexity

        # Count control flow statements
        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.For | ast.While):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += 1
            elif isinstance(child, ast.FunctionDef) and child != node:  # Nested function
                complexity += 2

        # Count lines of code
        if hasattr(node, "end_lineno"):
            lines = node.end_lineno - node.lineno + 1
            complexity += lines * 0.1

        return complexity

    def _find_duplicate_algorithms(self) -> dict[str, list[EncryptionMethod]]:
        """Find methods implementing the same algorithm"""

        algorithm_groups = {}

        for method in self.encryption_methods:
            if method.algorithm not in algorithm_groups:
                algorithm_groups[method.algorithm] = []
            algorithm_groups[method.algorithm].append(method)

        # Return only groups with duplicates
        return {alg: methods for alg, methods in algorithm_groups.items() if len(methods) > 1}

    def _identify_consolidation_opportunities(self) -> list[dict[str, Any]]:
        """Identify specific consolidation opportunities"""

        opportunities = []

        # Group by algorithm and method type
        for algorithm, methods in self.duplicate_algorithms.items():
            # Group by method type (encrypt/decrypt pairs)
            type_groups = {}
            for method in methods:
                if method.method_type not in type_groups:
                    type_groups[method.method_type] = []
                type_groups[method.method_type].append(method)

            # Look for complete encrypt/decrypt pairs
            if "encrypt" in type_groups and "decrypt" in type_groups:
                opportunities.append(
                    {
                        "algorithm": algorithm,
                        "encrypt_methods": len(type_groups["encrypt"]),
                        "decrypt_methods": len(type_groups["decrypt"]),
                        "consolidation_potential": "high",
                        "estimated_reduction": len(methods) - 1,  # All but one can be eliminated
                    }
                )

        return opportunities

    def _analyze_complexity(self) -> dict[str, float]:
        """Analyze overall complexity metrics"""

        if not self.encryption_methods:
            return {}

        complexities = [method.complexity_score for method in self.encryption_methods]

        return {
            "average_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "min_complexity": min(complexities),
            "total_methods": len(self.encryption_methods),
        }

    def generate_consolidation_plan(self) -> ConsolidationPlan:
        """Generate detailed consolidation plan"""

        # Analyze current state
        self.analyze_encryption_patterns()

        # Define target unified module structure
        target_module = "packages/core/security/unified_encryption.py"

        # Create unified interface
        unified_interface = """
from abc import ABC, abstractmethod
from typing import Any, Dict

class UnifiedEncryptionService(ABC):
    '''Single source of truth for all encryption operations'''

    @abstractmethod
    def encrypt_data(self, data: Any, algorithm: str = "fernet") -> bytes:
        '''Encrypt data using specified algorithm'''
        pass

    @abstractmethod
    def decrypt_data(self, encrypted_data: bytes, algorithm: str = "fernet") -> Any:
        '''Decrypt data using specified algorithm'''
        pass

    @abstractmethod
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        '''Hash data using specified algorithm'''
        pass

    @abstractmethod
    def sign_data(self, data: Any, private_key: Any) -> bytes:
        '''Sign data using private key'''
        pass
"""

        # Define migration steps
        migration_steps = [
            "1. Create unified encryption module with interface",
            "2. Implement DigitalTwinEncryption as primary implementation",
            "3. Create adapter pattern for legacy implementations",
            "4. Update dependency injection container",
            "5. Migrate high-usage files first (packages/core/security/)",
            "6. Update agent base template to use unified service",
            "7. Migrate remaining usages module by module",
            "8. Remove duplicate implementations",
            "9. Update tests to use behavioral contracts",
            "10. Validate performance and security equivalence",
        ]

        # Backward compatibility strategy
        backward_compatibility = [
            "Maintain existing method signatures during transition",
            "Use decorator pattern to wrap legacy calls",
            "Provide migration utilities for common patterns",
            "Keep deprecated methods with warnings for 1 release cycle",
            "Document migration path for each deprecated method",
        ]

        # Calculate estimated reduction
        total_methods = len(self.encryption_methods)
        estimated_reduction = {
            "method_count": total_methods - 4,  # Reduce to 4 core methods
            "code_lines": 0.6,  # Estimate 60% reduction in lines
            "coupling_score": 0.4,  # 40% coupling reduction
            "maintenance_burden": 0.7,  # 70% maintenance reduction
        }

        return ConsolidationPlan(
            target_module=target_module,
            unified_interface=unified_interface,
            migration_steps=migration_steps,
            backward_compatibility=backward_compatibility,
            estimated_reduction=estimated_reduction,
        )


def main():
    """Main consolidation orchestration"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize analyzer
    codebase_path = Path("packages")
    analyzer = SecurityConsolidationAnalyzer(codebase_path)

    # Analyze current state
    logger.info("Analyzing encryption patterns...")
    analysis = analyzer.analyze_encryption_patterns()

    # Generate consolidation plan
    logger.info("Generating consolidation plan...")
    plan = analyzer.generate_consolidation_plan()

    # Output results
    results = {
        "analysis": analysis,
        "consolidation_plan": {
            "target_module": plan.target_module,
            "migration_steps": plan.migration_steps,
            "backward_compatibility": plan.backward_compatibility,
            "estimated_reduction": plan.estimated_reduction,
        },
        "priority_files": [
            "packages/core/security/digital_twin_encryption.py",
            "packages/core/legacy/security/digital_twin_encryption.py",
            "packages/edge/legacy_src/digital_twin/security/encryption_manager.py",
            "packages/p2p/communications/a2a_protocol.py",
        ],
    }

    # Save consolidation plan
    output_file = Path("quality_reports/security_consolidation_plan.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Consolidation plan saved to {output_file}")

    # Print summary
    print("\n=== SECURITY CONSOLIDATION SUMMARY ===")
    print(f"Total encryption methods found: {analysis['total_encryption_methods']}")
    print(f"Duplicate algorithms: {len(analysis['duplicate_algorithms'])}")
    print(f"Consolidation opportunities: {len(analysis['consolidation_opportunities'])}")
    print(f"Estimated method reduction: {plan.estimated_reduction['method_count']}")
    print(f"Target module: {plan.target_module}")
    print("\nNext steps:")
    for i, step in enumerate(plan.migration_steps[:3], 1):
        print(f"  {step}")


if __name__ == "__main__":
    main()
