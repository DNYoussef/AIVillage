# Magic Literals Extraction Plan - Task Management Components

## Executive Summary

Analysis of the task_management components revealed **159 magic literals** across 6 files, with high coupling scores indicating significant improvement opportunities. This plan provides a comprehensive strategy for extracting all hardcoded values into type-safe, configurable constants.

## Analysis Results

### Coupling Metrics Overview
- **Total Files Analyzed**: 6
- **Total Magic Literals**: 159
- **Magic Literal Density**: 24.42 per 100 LOC (High)
- **Worst Coupled Files**:
  1. `unified_task_manager.py` (21.6 coupling score)
  2. `incentive_model.py` (14.95 coupling score)
  3. `__init__.py` (17.0 coupling score)

### Magic Literals by Category

#### 1. Timing and Intervals (18 instances)
```python
# Current magic literals
await asyncio.sleep(1)          # Line 310, unified_task_manager.py
learning_rate: float = 0.01     # Line 19, incentive_model.py
history_length: int = 1000      # Line 20, incentive_model.py
```

#### 2. Calculation Parameters (35 instances)
```python
# Performance adjustment factors
current * 1.1                   # Line 117, incentive_model.py  
current * 0.9                   # Line 119, incentive_model.py
max(current * 1.1, 2.0)        # Line 117, incentive_model.py
max(current * 0.9, 0.5)        # Line 119, incentive_model.py

# Incentive calculation factors
0.1 * reward                    # Lines 94, 100, 104, incentive_model.py
0.5 if result.get(...)          # Line 157, incentive_model.py
0.3 * len(result.get(...))      # Line 158, incentive_model.py
```

#### 3. Default Values and Thresholds (42 instances)
```python
# Task defaults
priority: int = 1               # Line 22, task.py; Line 59, unified_task_manager.py
self.batch_size = 5            # Line 53, unified_task_manager.py
progress: float = 0.0          # Line 30, unified_task_manager.py

# ML model defaults
n_components=5                  # Line 33, incentive_model.py
task_priority > 5              # Line 143, incentive_model.py
task_complexity > 7            # Line 147, incentive_model.py
```

#### 4. Status Strings and Identifiers (28 instances)
```python
# Project statuses
status: str = "initialized"     # Line 29, unified_task_manager.py

# Task statuses (already enum-based - good example)
PENDING = "pending"            # Line 8, task.py
IN_PROGRESS = "in_progress"    # Line 9, task.py
COMPLETED = "completed"        # Line 10, task.py
FAILED = "failed"             # Line 11, task.py

# Task type strings
task_type == "critical"        # Line 141, incentive_model.py
task_type == "routine"         # Line 143, incentive_model.py
"analysis" in task.get(...)    # Line 145, incentive_model.py
```

#### 5. Dictionary Keys and Field Names (36 instances)
```python
# Task dictionary keys
task["assigned_agent"]         # Multiple lines
task["collaborators"]          # Line 97, incentive_model.py
result.get("success", False)   # Line 114, incentive_model.py
result.get("quality", 0.5)     # Line 155, incentive_model.py

# Analysis result keys
"agent_id": task["assigned_agent"]     # Line 70, incentive_model.py
"incentive": float(adjusted_incentive) # Line 71, incentive_model.py
```

## Constants Module Design

### 1. Core Constants Structure

```python
# experiments/agents/agents/task_management/constants.py

from enum import Enum, IntEnum
from typing import Final, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConstants:
    """Task management constants."""
    
    # Batch processing
    DEFAULT_BATCH_SIZE: Final[int] = 5
    BATCH_PROCESSING_INTERVAL: Final[float] = 1.0  # seconds
    MAX_BATCH_SIZE: Final[int] = 100
    MIN_BATCH_SIZE: Final[int] = 1
    
    # Task defaults
    DEFAULT_PRIORITY: Final[int] = 1
    MAX_PRIORITY: Final[int] = 10
    MIN_PRIORITY: Final[int] = 1
    
    # Performance thresholds
    HIGH_PRIORITY_THRESHOLD: Final[int] = 5
    HIGH_COMPLEXITY_THRESHOLD: Final[int] = 7
    MIN_PERFORMANCE_HISTORY_SIZE: Final[int] = 2


@dataclass(frozen=True)
class IncentiveConstants:
    """Incentive model calculation constants."""
    
    # Learning parameters
    DEFAULT_LEARNING_RATE: Final[float] = 0.01
    MAX_LEARNING_RATE: Final[float] = 1.0
    MIN_LEARNING_RATE: Final[float] = 0.001
    
    # History management
    DEFAULT_HISTORY_LENGTH: Final[int] = 1000
    MIN_HISTORY_LENGTH: Final[int] = 10
    MAX_HISTORY_LENGTH: Final[int] = 10000
    
    # Performance adjustment factors
    PERFORMANCE_INCREASE_FACTOR: Final[float] = 1.1
    PERFORMANCE_DECREASE_FACTOR: Final[float] = 0.9
    MAX_PERFORMANCE_MULTIPLIER: Final[float] = 2.0
    MIN_PERFORMANCE_MULTIPLIER: Final[float] = 0.5
    
    # Reward calculation factors
    BASE_REWARD_MULTIPLIER: Final[int] = 10
    SPECIALIZATION_UPDATE_RATE: Final[float] = 0.1
    COLLABORATION_UPDATE_RATE: Final[float] = 0.1
    INNOVATION_UPDATE_RATE: Final[float] = 0.1
    
    # Incentive adjustment factors
    INNOVATION_BONUS_FACTOR: Final[float] = 0.5
    COLLABORATION_FACTOR: Final[float] = 0.3
    DEFAULT_QUALITY_FACTOR: Final[float] = 0.5
    
    # Calculation parameters
    DIFFICULTY_NORMALIZATION_DIVISOR: Final[int] = 100
    SKILL_WEIGHT_FACTOR: Final[float] = 0.1
    REWARD_AVERAGING_DIVISOR: Final[int] = 3
    
    # Trend analysis
    MIN_TREND_FACTOR: Final[float] = 0.5
    MAX_TREND_FACTOR: Final[float] = 1.5
    BASELINE_TREND_FACTOR: Final[float] = 1.0
    
    # PCA analysis
    DEFAULT_PCA_COMPONENTS: Final[int] = 5


@dataclass(frozen=True)
class ProjectConstants:
    """Project management constants."""
    
    # Default values
    DEFAULT_PROGRESS: Final[float] = 0.0
    COMPLETE_PROGRESS: Final[float] = 1.0
    DEFAULT_NAME: Final[str] = ""
    DEFAULT_DESCRIPTION: Final[str] = ""
    
    # Agent selection
    DEFAULT_AGENT_NAME: Final[str] = "default_agent"
    DECISION_THRESHOLD: Final[float] = 0.5
    
    # Array indexing
    FIRST_AGENT_INDEX: Final[int] = 0


class ProjectStatus(Enum):
    """Project status enumeration."""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskType(Enum):
    """Task type enumeration."""
    CRITICAL = "critical"
    ROUTINE = "routine"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    DEFAULT = "default"


class TaskDictKeys:
    """Task dictionary key constants."""
    ASSIGNED_AGENT: Final[str] = "assigned_agent"
    TASK_ID: Final[str] = "task_id"
    DESCRIPTION: Final[str] = "description"
    COLLABORATORS: Final[str] = "collaborators"
    TYPE: Final[str] = "type"
    PRIORITY: Final[str] = "priority"
    COMPLEXITY: Final[str] = "complexity"
    ESTIMATED_TIME: Final[str] = "estimated_time"
    REQUIRED_SKILLS: Final[str] = "required_skills"


class ResultDictKeys:
    """Result dictionary key constants."""
    SUCCESS: Final[str] = "success"
    TIME_TAKEN: Final[str] = "time_taken"
    EXPECTED_TIME: Final[str] = "expected_time"
    QUALITY: Final[str] = "quality"
    COST: Final[str] = "cost"
    BUDGET: Final[str] = "budget"
    INNOVATIVE_SOLUTION: Final[str] = "innovative_solution"
    COLLABORATORS: Final[str] = "collaborators"


class AnalysisResultKeys:
    """Analysis result dictionary key constants."""
    AGENT_ID: Final[str] = "agent_id"
    INCENTIVE: Final[str] = "incentive"
    AVERAGE: Final[str] = "average"
    TREND: Final[str] = "trend"
    LONG_TERM: Final[str] = "long_term"
    SPECIALIZATION: Final[str] = "specialization"
    COLLABORATION: Final[str] = "collaboration"
    INNOVATION: Final[str] = "innovation"
    PCA_COMPONENTS: Final[str] = "pca_components"


class MessageKeys:
    """Message protocol key constants."""
    TASK_ID: Final[str] = "task_id"
    DESCRIPTION: Final[str] = "description"
    INCENTIVE: Final[str] = "incentive"
    AGENT: Final[str] = "agent"


class CheckpointKeys:
    """Model checkpoint key constants."""
    INCENTIVE_MATRIX: Final[str] = "incentive_matrix"
    NUM_AGENTS: Final[str] = "num_agents"
    NUM_ACTIONS: Final[str] = "num_actions"
    LEARNING_RATE: Final[str] = "learning_rate"
    PERFORMANCE_HISTORY: Final[str] = "performance_history"
    TASK_DIFFICULTY_HISTORY: Final[str] = "task_difficulty_history"
    LONG_TERM_PERFORMANCE: Final[str] = "long_term_performance"
    AGENT_SPECIALIZATION: Final[str] = "agent_specialization"
    COLLABORATION_SCORE: Final[str] = "collaboration_score"
    INNOVATION_SCORE: Final[str] = "innovation_score"


class ActionMapping(IntEnum):
    """Action ID mapping for task types."""
    CRITICAL = 0
    HIGH_PRIORITY_ROUTINE = 1
    ANALYSIS = 2
    HIGH_COMPLEXITY = 3
    DEFAULT = 4
```

### 2. Configuration Override System

```python
# experiments/agents/agents/task_management/config.py

from typing import Dict, Any, Optional
from pathlib import Path
import json
import os
from dataclasses import asdict, replace

from .constants import TaskConstants, IncentiveConstants, ProjectConstants


class TaskManagementConfig:
    """Configuration manager with environment and file overrides."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.task_constants = TaskConstants()
        self.incentive_constants = IncentiveConstants()
        self.project_constants = ProjectConstants()
        
        # Load from file if provided
        if config_path and config_path.exists():
            self._load_from_file(config_path)
        
        # Apply environment overrides
        self._apply_environment_overrides()
    
    def _load_from_file(self, config_path: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update constants with file values
            if 'task' in config_data:
                self.task_constants = replace(
                    self.task_constants,
                    **config_data['task']
                )
            
            if 'incentive' in config_data:
                self.incentive_constants = replace(
                    self.incentive_constants,
                    **config_data['incentive']
                )
            
            if 'project' in config_data:
                self.project_constants = replace(
                    self.project_constants,
                    **config_data['project']
                )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Task constants
        batch_size = os.getenv('AIVILLAGE_BATCH_SIZE')
        if batch_size and batch_size.isdigit():
            self.task_constants = replace(
                self.task_constants,
                DEFAULT_BATCH_SIZE=int(batch_size)
            )
        
        # Incentive constants
        learning_rate = os.getenv('AIVILLAGE_LEARNING_RATE')
        if learning_rate:
            try:
                self.incentive_constants = replace(
                    self.incentive_constants,
                    DEFAULT_LEARNING_RATE=float(learning_rate)
                )
            except ValueError:
                pass
    
    def validate(self) -> bool:
        """Validate configuration values are within acceptable ranges."""
        try:
            # Validate task constants
            assert self.task_constants.MIN_BATCH_SIZE <= self.task_constants.DEFAULT_BATCH_SIZE <= self.task_constants.MAX_BATCH_SIZE
            assert self.task_constants.MIN_PRIORITY <= self.task_constants.DEFAULT_PRIORITY <= self.task_constants.MAX_PRIORITY
            
            # Validate incentive constants
            assert self.incentive_constants.MIN_LEARNING_RATE <= self.incentive_constants.DEFAULT_LEARNING_RATE <= self.incentive_constants.MAX_LEARNING_RATE
            assert self.incentive_constants.MIN_HISTORY_LENGTH <= self.incentive_constants.DEFAULT_HISTORY_LENGTH <= self.incentive_constants.MAX_HISTORY_LENGTH
            
            return True
        except AssertionError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'task': asdict(self.task_constants),
            'incentive': asdict(self.incentive_constants),
            'project': asdict(self.project_constants)
        }
    
    def save_to_file(self, config_path: Path) -> None:
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance
_config = TaskManagementConfig()

def get_config() -> TaskManagementConfig:
    """Get the global configuration instance."""
    return _config

def reload_config(config_path: Optional[Path] = None) -> None:
    """Reload configuration from file."""
    global _config
    _config = TaskManagementConfig(config_path)
```

## Migration Strategy

### Phase 1: Create Constants Infrastructure
1. **Create constants module** with all identified literals
2. **Create configuration system** with override capabilities
3. **Add validation** for constant values
4. **Create tests** for constants and configuration

### Phase 2: Incremental Replacement
1. **Start with unified_task_manager.py** (highest coupling)
   - Replace batch_size with `config.task_constants.DEFAULT_BATCH_SIZE`
   - Replace sleep(1) with `config.task_constants.BATCH_PROCESSING_INTERVAL`
   - Replace string literals with enum values

2. **Continue with incentive_model.py**
   - Replace all calculation factors with named constants
   - Replace dictionary keys with constant definitions
   - Update ML parameters to use configuration

3. **Complete remaining files**
   - Task.py is already well-structured with enums
   - Update remaining files following the same pattern

### Phase 3: Configuration Integration
1. **Add configuration loading** to module initialization
2. **Implement environment variable overrides**
3. **Create configuration validation**
4. **Add runtime configuration updates** where appropriate

### Phase 4: Testing and Validation
1. **Unit tests** for all constants usage
2. **Integration tests** with different configurations
3. **Performance validation** to ensure no regression
4. **Backwards compatibility verification**

## Implementation Examples

### Before and After Comparisons

#### 1. Batch Processing (unified_task_manager.py)
```python
# BEFORE: Magic literals
class UnifiedManagement:
    def __init__(self, ...):
        self.batch_size = 5  # Magic number
    
    async def start_batch_processing(self):
        while True:
            await self.process_task_batch()
            await asyncio.sleep(1)  # Magic number

# AFTER: Type-safe constants
from .constants import get_config

class UnifiedManagement:
    def __init__(self, ...):
        self._config = get_config()
        self.batch_size = self._config.task_constants.DEFAULT_BATCH_SIZE
    
    async def start_batch_processing(self):
        while True:
            await self.process_task_batch()
            await asyncio.sleep(self._config.task_constants.BATCH_PROCESSING_INTERVAL)
```

#### 2. Incentive Calculations (incentive_model.py)
```python
# BEFORE: Magic literals
def update_agent_performance(self, agent_performance, agent, result, analytics):
    success = result.get("success", False)  # Magic string
    current = agent_performance.get(agent, 1.0)  # Magic number
    if success:
        agent_performance[agent] = min(current * 1.1, 2.0)  # Magic numbers
    else:
        agent_performance[agent] = max(current * 0.9, 0.5)  # Magic numbers

# AFTER: Type-safe constants
from .constants import get_config, ResultDictKeys

def update_agent_performance(self, agent_performance, agent, result, analytics):
    config = get_config()
    success = result.get(ResultDictKeys.SUCCESS, False)
    current = agent_performance.get(agent, config.incentive_constants.BASELINE_TREND_FACTOR)
    if success:
        agent_performance[agent] = min(
            current * config.incentive_constants.PERFORMANCE_INCREASE_FACTOR,
            config.incentive_constants.MAX_PERFORMANCE_MULTIPLIER
        )
    else:
        agent_performance[agent] = max(
            current * config.incentive_constants.PERFORMANCE_DECREASE_FACTOR,
            config.incentive_constants.MIN_PERFORMANCE_MULTIPLIER
        )
```

#### 3. Task Type Mapping (incentive_model.py)
```python
# BEFORE: Magic strings and numbers
def _map_task_to_action(self, task):
    task_type = task.get("type", "default")
    if task_type == "critical":
        return 0
    if task_type == "routine" and task.get("priority", 1) > 5:
        return 1
    if "analysis" in task.get("description", "").lower():
        return 2
    return 4 % self.num_actions

# AFTER: Type-safe enums and constants
from .constants import TaskType, TaskDictKeys, ActionMapping, get_config

def _map_task_to_action(self, task):
    config = get_config()
    task_type = TaskType(task.get(TaskDictKeys.TYPE, TaskType.DEFAULT.value))
    task_priority = task.get(TaskDictKeys.PRIORITY, config.task_constants.DEFAULT_PRIORITY)
    
    if task_type == TaskType.CRITICAL:
        return ActionMapping.CRITICAL
    if task_type == TaskType.ROUTINE and task_priority > config.task_constants.HIGH_PRIORITY_THRESHOLD:
        return ActionMapping.HIGH_PRIORITY_ROUTINE
    if TaskType.ANALYSIS.value in task.get(TaskDictKeys.DESCRIPTION, "").lower():
        return ActionMapping.ANALYSIS
    if task.get(TaskDictKeys.COMPLEXITY, 1) > config.task_constants.HIGH_COMPLEXITY_THRESHOLD:
        return ActionMapping.HIGH_COMPLEXITY
    return ActionMapping.DEFAULT
```

## Configuration Examples

### 1. Default Configuration File
```json
{
  "task": {
    "DEFAULT_BATCH_SIZE": 5,
    "BATCH_PROCESSING_INTERVAL": 1.0,
    "DEFAULT_PRIORITY": 1,
    "HIGH_PRIORITY_THRESHOLD": 5,
    "HIGH_COMPLEXITY_THRESHOLD": 7
  },
  "incentive": {
    "DEFAULT_LEARNING_RATE": 0.01,
    "DEFAULT_HISTORY_LENGTH": 1000,
    "PERFORMANCE_INCREASE_FACTOR": 1.1,
    "PERFORMANCE_DECREASE_FACTOR": 0.9,
    "MAX_PERFORMANCE_MULTIPLIER": 2.0,
    "MIN_PERFORMANCE_MULTIPLIER": 0.5,
    "DEFAULT_PCA_COMPONENTS": 5
  },
  "project": {
    "DEFAULT_PROGRESS": 0.0,
    "DECISION_THRESHOLD": 0.5
  }
}
```

### 2. Environment Variable Overrides
```bash
# High-throughput configuration
export AIVILLAGE_BATCH_SIZE=20
export AIVILLAGE_LEARNING_RATE=0.001

# Development configuration
export AIVILLAGE_BATCH_SIZE=1
export AIVILLAGE_LEARNING_RATE=0.1
```

## Backwards Compatibility

### 1. Gradual Migration Support
```python
# Support both old and new patterns during migration
class UnifiedManagement:
    def __init__(self, ...):
        # Try new configuration first, fallback to old values
        try:
            config = get_config()
            self.batch_size = config.task_constants.DEFAULT_BATCH_SIZE
        except ImportError:
            self.batch_size = 5  # Legacy fallback
    
    def set_batch_size(self, size: int) -> None:
        """Maintain existing API for backwards compatibility."""
        self.batch_size = size
```

### 2. Deprecation Warnings
```python
import warnings
from typing import Optional

def legacy_method(magic_value: Optional[int] = None):
    if magic_value is not None:
        warnings.warn(
            "Passing magic values is deprecated. Use configuration instead.",
            DeprecationWarning,
            stacklevel=2
        )
    # Use configuration or provided value
    value = magic_value if magic_value is not None else get_config().task_constants.DEFAULT_BATCH_SIZE
```

## Testing Strategy

### 1. Constants Validation Tests
```python
# tests/task_management/test_constants.py
import pytest
from experiments.agents.agents.task_management.constants import TaskConstants, IncentiveConstants

class TestConstants:
    def test_task_constants_validation(self):
        constants = TaskConstants()
        assert constants.MIN_BATCH_SIZE <= constants.DEFAULT_BATCH_SIZE <= constants.MAX_BATCH_SIZE
        assert constants.MIN_PRIORITY <= constants.DEFAULT_PRIORITY <= constants.MAX_PRIORITY
    
    def test_incentive_constants_validation(self):
        constants = IncentiveConstants()
        assert 0 < constants.DEFAULT_LEARNING_RATE < 1
        assert constants.MIN_PERFORMANCE_MULTIPLIER < constants.MAX_PERFORMANCE_MULTIPLIER
```

### 2. Configuration Override Tests
```python
# tests/task_management/test_config.py
import tempfile
import json
from pathlib import Path
from experiments.agents.agents.task_management.config import TaskManagementConfig

class TestConfiguration:
    def test_file_override(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"task": {"DEFAULT_BATCH_SIZE": 10}}
            json.dump(config_data, f)
            f.flush()
            
            config = TaskManagementConfig(Path(f.name))
            assert config.task_constants.DEFAULT_BATCH_SIZE == 10
    
    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv('AIVILLAGE_BATCH_SIZE', '15')
        config = TaskManagementConfig()
        assert config.task_constants.DEFAULT_BATCH_SIZE == 15
```

### 3. Migration Integration Tests
```python
# tests/task_management/test_migration.py
class TestMigration:
    def test_unified_manager_uses_constants(self):
        """Test that UnifiedManagement uses configuration constants."""
        manager = UnifiedManagement(...)
        config = get_config()
        assert manager.batch_size == config.task_constants.DEFAULT_BATCH_SIZE
    
    def test_incentive_model_uses_constants(self):
        """Test that IncentiveModel uses configuration constants."""
        model = IncentiveModel(10, 5)
        config = get_config()
        assert model.learning_rate == config.incentive_constants.DEFAULT_LEARNING_RATE
```

## Performance Considerations

### 1. Configuration Caching
- Use singleton pattern for configuration instance
- Cache expensive calculations based on constants
- Lazy load configuration values where appropriate

### 2. Import Optimization
```python
# Avoid circular imports by using late binding
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .constants import TaskManagementConfig

def get_batch_size() -> int:
    from .constants import get_config
    return get_config().task_constants.DEFAULT_BATCH_SIZE
```

## Success Metrics

### 1. Quantitative Goals
- **Reduce magic literal count from 159 to < 10**
- **Improve coupling scores**:
  - unified_task_manager.py: 21.6 → < 15
  - incentive_model.py: 14.95 → < 10
  - Overall average: 9.6 → < 7
- **Maintain 100% test coverage**
- **Zero performance regression**

### 2. Qualitative Improvements
- **Type safety**: All literals replaced with typed constants
- **Configurability**: Runtime configuration without code changes
- **Maintainability**: Clear parameter meanings and relationships
- **Testability**: Easy to test with different configuration values

## Timeline and Dependencies

### Week 1: Infrastructure
- Create constants and configuration modules
- Write initial tests
- Set up CI integration

### Week 2: Core Migration
- Migrate unified_task_manager.py
- Migrate incentive_model.py
- Update tests

### Week 3: Integration
- Complete remaining files
- Add environment variable support
- Performance validation

### Week 4: Documentation and Finalization
- Update documentation
- Code review and refinement
- Deployment preparation

## Conclusion

This comprehensive plan addresses all 159 magic literals found in the task_management components, providing a type-safe, configurable, and maintainable solution. The phased approach ensures backwards compatibility while significantly improving code quality and reducing coupling scores.

The implementation will transform hard-to-maintain magic literals into a robust configuration system that supports:
- Type safety through enums and typed constants
- Runtime configuration via files and environment variables
- Easy testing with different parameter combinations
- Clear documentation of parameter meanings and constraints

Success will be measured by quantitative coupling improvements and qualitative code quality enhancements, making the task management system more maintainable and reliable.