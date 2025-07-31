# AIVillage Code Cleanup Roadmap

**Sprint 4 Quality Improvement Plan**  
**Focus:** Critical technical debt resolution and production hardening

## 🎯 Immediate Actions (This Sprint)

### Priority 1: MCP Server Infrastructure Completion

#### File: `mcp_servers/hyperag/protocol.py`
**Issues:** 5 critical TODO items blocking core functionality

```python
# BEFORE (Current TODO items):
async def retrieve_information(self, query: str, filters: dict = None) -> dict:
    # TODO: Implement actual retrieval and reasoning
    return {"placeholder": "not implemented"}

# AFTER (Implementation needed):
async def retrieve_information(self, query: str, filters: dict = None) -> dict:
    """Retrieve and process information using hybrid retrieval system."""
    try:
        # Use existing retrieval components
        retriever = HybridRetriever(self.vector_store, self.graph_store)
        results = await retriever.retrieve(query, filters or {})
        
        # Apply reasoning layer
        reasoning_engine = ReasoningEngine(self.model_config)
        processed_results = await reasoning_engine.process(results)
        
        return {
            "query": query,
            "results": processed_results,
            "confidence": processed_results.confidence_score,
            "metadata": processed_results.metadata
        }
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise MCP_Error(f"Failed to retrieve information: {e}")
```

**Effort Estimate:** 2-3 days  
**Impact:** Enables full MCP server functionality

#### File: `mcp_servers/hyperag/memory/hippo_index.py`
**Issues:** Memory management not implemented

```python
# BEFORE:
memory_usage_mb=0.0,  # TODO: Calculate actual usage
pending_consolidations=0,  # TODO: Implement

# AFTER:
def calculate_memory_usage(self) -> float:
    """Calculate actual memory usage of the index."""
    import psutil
    import sys
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Estimate index-specific memory usage
    index_memory = sum(
        sys.getsizeof(self.embeddings),
        sys.getsizeof(self.metadata),
        sys.getsizeof(self.connections)
    ) / (1024 * 1024)  # Convert to MB
    
    return index_memory

def get_pending_consolidations(self) -> int:
    """Get count of pending consolidation tasks."""
    if not hasattr(self, '_consolidation_queue'):
        self._consolidation_queue = []
    return len(self._consolidation_queue)
```

**Effort Estimate:** 1 day  
**Impact:** Proper resource monitoring

### Priority 2: Error Handling Standardization

Create a common error handling utility to eliminate code duplication:

```python
# File: agent_forge/utils/error_handling.py
"""Standardized error handling patterns for AIVillage."""

import functools
import logging
from typing import Any, Callable, Optional, Type, Union

logger = logging.getLogger(__name__)

class AIVillageError(Exception):
    """Base exception for AIVillage errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

class ConfigurationError(AIVillageError):
    """Raised when configuration is invalid."""
    pass

class ProcessingError(AIVillageError):
    """Raised when processing fails."""
    pass

def handle_errors(
    *exception_types: Type[Exception],
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False
):
    """Decorator for standardized error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_types as e:
                log_method = getattr(logger, log_level)
                log_method(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                return default_return
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                log_method = getattr(logger, log_level)
                log_method(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Usage example:
@handle_errors(ProcessingError, ValueError, default_return={}, log_level="warning")
async def process_data(data):
    # Processing logic here
    pass
```

### Priority 3: Configuration Management Consolidation

Create centralized configuration to eliminate duplication:

```python
# File: agent_forge/config/base_config.py
"""Centralized configuration management."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

class BaseConfig(BaseModel):
    """Base configuration class with common patterns."""
    
    # Common paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    
    # Common model settings
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    precision: str = Field(default="float16")
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = None
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "BaseConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)

class ModelConfig(BaseConfig):
    """Model-specific configuration."""
    model_name: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class ServerConfig(BaseConfig):
    """Server configuration."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
```

## 🔧 Phase 2: Code Quality Improvements (Next Sprint)

### Complexity Reduction Targets

#### File: `agent_forge/forge_orchestrator.py`
**Current Issues:**
- Large class (300+ lines)
- Multiple responsibilities
- High cyclomatic complexity

**Refactoring Plan:**

```python
# BEFORE: Monolithic orchestrator
class ForgeOrchestrator:
    def __init__(self):
        # 50+ lines of initialization
        pass
    
    def run_pipeline(self):
        # 100+ lines doing everything
        pass

# AFTER: Decomposed orchestrator
class ForgeOrchestrator:
    """Main orchestrator - delegates to specialized components."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.workflow_manager = WorkflowManager(config.workflow)
        self.phase_executor = PhaseExecutor(config.execution)
        self.result_aggregator = ResultAggregator(config.reporting)
    
    async def run_pipeline(self, pipeline_spec: PipelineSpec) -> PipelineResult:
        """Execute pipeline using specialized components."""
        workflow = await self.workflow_manager.create_workflow(pipeline_spec)
        results = await self.phase_executor.execute_phases(workflow)
        return await self.result_aggregator.aggregate_results(results)

class WorkflowManager:
    """Handles workflow creation and validation."""
    
    async def create_workflow(self, spec: PipelineSpec) -> Workflow:
        # Focused on workflow logic only
        pass

class PhaseExecutor:
    """Executes individual pipeline phases."""
    
    async def execute_phases(self, workflow: Workflow) -> List[PhaseResult]:
        # Focused on execution logic only
        pass

class ResultAggregator:
    """Aggregates and formats results."""
    
    async def aggregate_results(self, results: List[PhaseResult]) -> PipelineResult:
        # Focused on result processing only
        pass
```

### Import Optimization

Create import optimization utility:

```python
# File: scripts/optimize_imports.py
"""Optimize imports across the codebase."""

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

def analyze_imports(file_path: Path) -> Dict[str, List[str]]:
    """Analyze imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    imports = defaultdict(list)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports['direct'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports[module].append(alias.name)
    
    return dict(imports)

def suggest_optimizations(import_analysis: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """Suggest import optimizations."""
    suggestions = []
    
    # Find consolidation opportunities
    for file_path, imports in import_analysis.items():
        module_counts = defaultdict(int)
        for module, names in imports.items():
            if module != 'direct':
                module_counts[module] += len(names)
        
        # Suggest consolidation for modules with many imports
        for module, count in module_counts.items():
            if count > 5:
                suggestions.append(
                    f"{file_path}: Consolidate {count} imports from {module}"
                )
    
    return suggestions
```

## 📚 Phase 3: Documentation Enhancement

### Documentation Templates

Create standardized templates for different component types:

```python
# File: docs/templates/module_template.py
"""Module Documentation Template.

This module provides [brief description of purpose].

Key Components:
    - [Component1]: [Brief description]
    - [Component2]: [Brief description]

Usage:
    Basic usage example:
    
    ```python
    from module import Component
    
    component = Component(config)
    result = component.process(data)
    ```

Note:
    [Any important notes or limitations]
"""

from typing import Any, Dict, List, Optional

class ExampleClass:
    """Example class following documentation standards.
    
    This class demonstrates the documentation standards for AIVillage.
    All public classes should follow this pattern.
    
    Attributes:
        config: Configuration object for the class.
        state: Current state of the instance.
    
    Example:
        Basic usage:
        
        ```python
        config = Config(param1="value1")
        instance = ExampleClass(config)
        result = instance.process_data(data)
        ```
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the example class.
        
        Args:
            config: Configuration dictionary containing:
                - param1: Description of param1
                - param2: Description of param2
                
        Raises:
            ConfigurationError: If required configuration is missing.
        """
        self.config = config
        self.state = "initialized"
    
    def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process input data and return results.
        
        This method processes the input data according to the configured
        parameters and returns structured results.
        
        Args:
            data: List of data items to process. Each item should contain:
                - id: Unique identifier
                - content: Content to process
                
        Returns:
            Dictionary containing:
                - processed_items: Number of items processed
                - results: List of processing results
                - errors: List of any errors encountered
                
        Raises:
            ProcessingError: If data processing fails.
            
        Example:
            ```python
            data = [{"id": 1, "content": "text"}]
            result = instance.process_data(data)
            print(f"Processed {result['processed_items']} items")
            ```
        """
        # Implementation here
        pass
```

### Documentation Coverage Improvement Script

```python
# File: scripts/improve_documentation.py
"""Automatically improve documentation coverage."""

import ast
import re
from pathlib import Path
from typing import List, Optional

class DocumentationImprover:
    """Improves documentation coverage by adding missing docstrings."""
    
    def __init__(self, target_directories: List[str]):
        self.target_directories = target_directories
    
    def analyze_missing_docs(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze file for missing documentation."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        missing_docs = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not self._has_docstring(node):
                    missing_docs.append({
                        'type': node.__class__.__name__,
                        'name': node.name,
                        'line': node.lineno,
                        'complexity': self._estimate_complexity(node)
                    })
        
        return missing_docs
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has a docstring."""
        if not node.body:
            return False
        
        first_stmt = node.body[0]
        return (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Str))
    
    def generate_docstring_template(self, node_info: Dict[str, Any]) -> str:
        """Generate docstring template based on node information."""
        if node_info['type'] == 'ClassDef':
            return f'"""{"[Brief description of the class]"}.\n    \n    {"[Detailed description]"}\n    """'
        else:
            return f'"""{"[Brief description of the function]"}.\n    \n    Args:\n        {"[Add arguments]"}\n    \n    Returns:\n        {"[Add return description]"}\n    """'
```

## 🧪 Phase 4: Testing & Quality Assurance

### Enhanced Testing Strategy

```python
# File: tests/quality/test_code_quality.py
"""Code quality tests to enforce standards."""

import ast
import pytest
from pathlib import Path
from typing import List

class TestCodeQuality:
    """Test suite for code quality enforcement."""
    
    @pytest.mark.parametrize("python_file", Path("production").rglob("*.py"))
    def test_production_no_todos(self, python_file: Path):
        """Ensure production code has no TODO items."""
        with open(python_file, 'r') as f:
            content = f.read()
        
        todo_pattern = r'(TODO|FIXME|XXX|HACK)'
        matches = re.findall(todo_pattern, content, re.IGNORECASE)
        
        assert not matches, f"Found TODO/FIXME in production file: {python_file}"
    
    @pytest.mark.parametrize("python_file", Path("production").rglob("*.py"))
    def test_production_docstring_coverage(self, python_file: Path):
        """Ensure production code has good docstring coverage."""
        with open(python_file, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        total_functions = 0
        documented_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                if self._has_docstring(node):
                    documented_functions += 1
        
        if total_functions > 0:
            coverage = documented_functions / total_functions
            assert coverage >= 0.8, f"Documentation coverage too low: {coverage:.2%}"
    
    def test_complexity_thresholds(self):
        """Test that code complexity stays within thresholds."""
        high_complexity_files = []
        
        for py_file in Path("production").rglob("*.py"):
            complexity = self._calculate_file_complexity(py_file)
            if complexity > 15:  # Threshold for production
                high_complexity_files.append((py_file, complexity))
        
        assert not high_complexity_files, f"High complexity files: {high_complexity_files}"
```

## 📊 Progress Tracking

### Quality Metrics Dashboard Updates

```python
# File: scripts/quality_dashboard_update.py
"""Update quality metrics dashboard."""

import json
from datetime import datetime
from pathlib import Path

def update_quality_metrics():
    """Update the quality metrics dashboard."""
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "todo_count": count_todos(),
        "documentation_coverage": calculate_doc_coverage(),
        "complexity_violations": find_complexity_violations(),
        "test_coverage": get_test_coverage()
    }
    
    # Save to dashboard file
    with open("quality_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate trend analysis
    generate_trend_report()

def generate_trend_report():
    """Generate trend analysis of quality metrics."""
    # Load historical data and analyze trends
    pass
```

## 🎯 Success Criteria

### Sprint 4 Targets
- [ ] **Complete MCP Server TODOs** (11 items → 0 items)
- [ ] **Implement error handling utility** (standardize across 20+ files)
- [ ] **Create configuration management** (consolidate 15+ config patterns)
- [ ] **Refactor 3 high-complexity files** (reduce complexity by 30%)
- [ ] **Improve documentation coverage** (core components to 80%+)

### Quality Gate Thresholds (Updated)
- **Production TODO Count:** 0 (maintain)
- **Core Infrastructure TODO Count:** <5 (from current 11)
- **Documentation Coverage:** 80%+ for production/core
- **Function Complexity:** <12 average for production
- **Security Issues:** 0 critical (maintain)

### Automated Quality Monitoring

```bash
# Daily quality check (to be added to CI/CD)
python scripts/daily_quality_check.py
python scripts/complexity_monitor.py --threshold=12
python scripts/todo_tracker.py --alert-threshold=5
```

---

**Next Review:** End of Sprint 4  
**Quality Champion:** Code Quality Agent  
**Implementation Priority:** MCP Server completion → Documentation → Refactoring