#!/usr/bin/env python3
"""
Constants Consolidation Agent - Magic Literals Elimination Specialist
Specialized agent for eliminating 159 magic literals with type-safe constants
Target: 159 hardcoded values → 0
"""

import asyncio
import ast
import re
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, IntEnum, StrEnum
import json

from ..agent_coordination_protocols import RefactoringAgent, AgentTask, TaskStatus, AgentType

class LiteralCategory(StrEnum):
    """Categories of magic literals"""
    TIMING = "timing"
    CALCULATION = "calculation"
    DEFAULT_VALUE = "default_value"
    STATUS_STRING = "status_string"
    CONFIGURATION_KEY = "configuration_key"
    VALIDATION_RULE = "validation_rule"
    MESSAGE_TEMPLATE = "message_template"
    THRESHOLD = "threshold"

@dataclass
class MagicLiteral:
    """Represents a magic literal found in code"""
    value: Union[str, int, float, bool]
    literal_type: str  # 'string', 'number', 'boolean'
    category: LiteralCategory
    locations: List[str] = field(default_factory=list)  # file:line locations
    usage_context: str = ""
    suggested_constant_name: str = ""
    replacement_priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class ConstantsGrouping:
    """Grouping of constants by category and module"""
    category: LiteralCategory
    module_name: str
    constants: Dict[str, MagicLiteral] = field(default_factory=dict)
    enum_name: Optional[str] = None
    interface_name: Optional[str] = None

class ConstantsConsolidatorAgent(RefactoringAgent):
    """Specialized agent for magic literals elimination"""
    
    def __init__(self, agent_type: AgentType, coordinator):
        super().__init__(agent_type, coordinator)
        self.found_literals: List[MagicLiteral] = []
        self.constants_groupings: List[ConstantsGrouping] = []
        self.generated_constants: Dict[str, str] = {}  # filename -> content
        self.task_management_files: List[Path] = []
        
    async def _prepare_phase_tasks(self):
        """Prepare Phase 1: Literal Discovery and Categorization"""
        tasks = [
            AgentTask(
                task_id="scan_magic_literals",
                agent_type=self.agent_type,
                description="Scan task_management files and identify all 159 magic literals",
                dependencies=[],
                outputs=["magic_literals_catalog", "literal_locations"]
            ),
            AgentTask(
                task_id="categorize_literals",
                agent_type=self.agent_type,
                description="Categorize literals by timing, calculations, defaults, etc.",
                dependencies=["magic_literals_catalog"],
                outputs=["categorized_literals", "constants_groupings"]
            ),
            AgentTask(
                task_id="design_constants_architecture",
                agent_type=self.agent_type,
                description="Design type-safe constants organization with enums",
                dependencies=["categorized_literals"],
                outputs=["constants_architecture", "enum_designs"]
            )
        ]
        
        for task in tasks:
            self.current_tasks.append(task)
            if self._all_dependencies_satisfied(task):
                await self._start_task(task)
                
    async def _start_implementation_tasks(self):
        """Implementation Phase: Constants Creation"""
        implementation_tasks = [
            AgentTask(
                task_id="generate_timing_constants",
                agent_type=self.agent_type,
                description="Create timing constants module with type-safe enums",
                dependencies=["constants_architecture"],
                outputs=["timing_constants_module"]
            ),
            AgentTask(
                task_id="generate_calculation_constants",
                agent_type=self.agent_type,
                description="Create calculation constants with mathematical enums",
                dependencies=["timing_constants_module"],
                outputs=["calculation_constants_module"]
            ),
            AgentTask(
                task_id="generate_status_constants",
                agent_type=self.agent_type,
                description="Create status string constants with string enums",
                dependencies=["calculation_constants_module"],
                outputs=["status_constants_module"]
            ),
            AgentTask(
                task_id="generate_configuration_constants",
                agent_type=self.agent_type,
                description="Create configuration constants with override system",
                dependencies=["status_constants_module"],
                outputs=["configuration_constants_module"]
            ),
            AgentTask(
                task_id="replace_literals_in_code",
                agent_type=self.agent_type,
                description="Replace all 159 literals with constant references",
                dependencies=["configuration_constants_module"],
                outputs=["refactored_files"]
            )
        ]
        
        for task in implementation_tasks:
            self.current_tasks.append(task)
            if self._all_dependencies_satisfied(task):
                await self._start_task(task)
                
    async def _begin_validation_tasks(self):
        """Validation Phase: Verify Complete Elimination"""
        validation_tasks = [
            AgentTask(
                task_id="validate_literal_elimination",
                agent_type=self.agent_type,
                description="Verify all 159 magic literals have been eliminated",
                dependencies=["refactored_files"],
                outputs=["elimination_validation_report"]
            ),
            AgentTask(
                task_id="validate_type_safety",
                agent_type=self.agent_type,
                description="Ensure all constants are type-safe and properly imported",
                dependencies=["elimination_validation_report"],
                outputs=["type_safety_validation"]
            )
        ]
        
        for task in validation_tasks:
            self.current_tasks.append(task)
            if self._all_dependencies_satisfied(task):
                await self._start_task(task)
                
    async def _execute_task(self, task: AgentTask):
        """Execute specific constants consolidation tasks"""
        try:
            if task.task_id == "scan_magic_literals":
                await self._scan_magic_literals(task)
            elif task.task_id == "categorize_literals":
                await self._categorize_literals(task)
            elif task.task_id == "design_constants_architecture":
                await self._design_constants_architecture(task)
            elif task.task_id == "generate_timing_constants":
                await self._generate_timing_constants(task)
            elif task.task_id == "generate_calculation_constants":
                await self._generate_calculation_constants(task)
            elif task.task_id == "generate_status_constants":
                await self._generate_status_constants(task)
            elif task.task_id == "generate_configuration_constants":
                await self._generate_configuration_constants(task)
            elif task.task_id == "replace_literals_in_code":
                await self._replace_literals_in_code(task)
            elif task.task_id == "validate_literal_elimination":
                await self._validate_literal_elimination(task)
            elif task.task_id == "validate_type_safety":
                await self._validate_type_safety(task)
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
    async def _scan_magic_literals(self, task: AgentTask):
        """Scan task_management files and identify all magic literals"""
        
        # Find task management files
        self.task_management_files = await self._find_task_management_files()
        
        if not self.task_management_files:
            self.logger.error("No task management files found")
            task.status = TaskStatus.FAILED
            return
            
        self.logger.info(f"Found {len(self.task_management_files)} task management files")
        
        # Scan each file for literals
        all_literals = []
        literal_locations = {}
        
        for file_path in self.task_management_files:
            literals = await self._extract_literals_from_file(file_path)
            all_literals.extend(literals)
            literal_locations[str(file_path)] = len(literals)
            
        self.found_literals = all_literals
        
        # Update metric - track progress toward 0 literals
        self.coordinator.memory_store.update_coupling_metric("MagicLiterals", len(all_literals))
        
        magic_literals_catalog = {
            "total_literals_found": len(all_literals),
            "target_elimination": 159,
            "files_scanned": len(self.task_management_files),
            "literal_types": self._count_by_type(all_literals),
            "priority_distribution": self._count_by_priority(all_literals)
        }
        
        outputs = {
            "magic_literals_catalog": magic_literals_catalog,
            "literal_locations": literal_locations
        }
        
        await self._complete_task(task, outputs)
        
    async def _categorize_literals(self, task: AgentTask):
        """Categorize literals by timing, calculations, defaults, etc."""
        
        categorized = {category: [] for category in LiteralCategory}
        
        for literal in self.found_literals:
            category = self._determine_category(literal)
            literal.category = category
            literal.suggested_constant_name = self._suggest_constant_name(literal)
            categorized[category].append(literal)
            
        # Create constants groupings
        self.constants_groupings = []
        for category, literals in categorized.items():
            if literals:
                grouping = ConstantsGrouping(
                    category=category,
                    module_name=f"{category.value}_constants",
                    constants={lit.suggested_constant_name: lit for lit in literals},
                    enum_name=f"{category.value.title()}Constants",
                    interface_name=f"I{category.value.title()}Config"
                )
                self.constants_groupings.append(grouping)
                
        categorization_results = {
            "categories_found": len([cat for cat, lits in categorized.items() if lits]),
            "category_breakdown": {cat.value: len(lits) for cat, lits in categorized.items()},
            "groupings_created": len(self.constants_groupings),
            "largest_category": max(categorized.keys(), key=lambda cat: len(categorized[cat])).value
        }
        
        outputs = {
            "categorized_literals": categorized,
            "constants_groupings": self.constants_groupings
        }
        
        await self._complete_task(task, outputs)
        
    async def _generate_timing_constants(self, task: AgentTask):
        """Create timing constants module with type-safe enums"""
        
        timing_grouping = next((g for g in self.constants_groupings 
                               if g.category == LiteralCategory.TIMING), None)
        
        if not timing_grouping:
            self.logger.warning("No timing constants found")
            await self._complete_task(task, {"timing_constants_module": ""})
            return
            
        timing_constants_code = f'''#!/usr/bin/env python3
"""
Timing Constants - Eliminating Magic Literals
Generated from magic literal extraction to improve code maintainability
"""

from typing import Dict, Any, Final
from enum import IntEnum, Enum
import os
from dataclasses import dataclass

class TimingConstants(IntEnum):
    """Type-safe timing constants with integer values"""
{self._generate_int_enum_values(timing_grouping)}

class TimingDelays(IntEnum):
    """Delay constants in milliseconds"""
{self._generate_delay_values(timing_grouping)}

class TimeoutValues(IntEnum):
    """Timeout constants in seconds"""
{self._generate_timeout_values(timing_grouping)}

@dataclass(frozen=True)
class TimingConfig:
    """Immutable timing configuration"""
{self._generate_timing_config_fields(timing_grouping)}

# Default timing configuration
DEFAULT_TIMING_CONFIG: Final[TimingConfig] = TimingConfig(
{self._generate_default_timing_config(timing_grouping)}
)

# Environment variable override support
def load_timing_config() -> TimingConfig:
    """Load timing configuration with environment variable overrides"""
    return TimingConfig(
{self._generate_env_override_config(timing_grouping)}
    )

# Runtime configuration
_timing_config: TimingConfig = load_timing_config()

def get_timing_config() -> TimingConfig:
    """Get current timing configuration"""
    return _timing_config

def update_timing_config(config: TimingConfig) -> None:
    """Update timing configuration at runtime"""
    global _timing_config
    _timing_config = config

# Convenience functions
def get_delay_ms(delay_type: str) -> int:
    """Get delay in milliseconds by type"""
    config = get_timing_config()
    return getattr(config, f"{delay_type}_delay", TimingDelays.DEFAULT_DELAY.value)

def get_timeout_seconds(timeout_type: str) -> int:
    """Get timeout in seconds by type"""
    config = get_timing_config()
    return getattr(config, f"{timeout_type}_timeout", TimeoutValues.DEFAULT_TIMEOUT.value)
'''
        
        self.generated_constants["timing_constants.py"] = timing_constants_code
        
        outputs = {
            "timing_constants_module": timing_constants_code
        }
        
        await self._complete_task(task, outputs)
        
    async def _generate_status_constants(self, task: AgentTask):
        """Create status string constants with string enums"""
        
        status_grouping = next((g for g in self.constants_groupings 
                               if g.category == LiteralCategory.STATUS_STRING), None)
        
        if not status_grouping:
            self.logger.warning("No status constants found")
            await self._complete_task(task, {"status_constants_module": ""})
            return
            
        status_constants_code = f'''#!/usr/bin/env python3
"""
Status Constants - Eliminating Magic String Literals
Type-safe status strings with validation and translation support
"""

from typing import Dict, Set, Final, Optional
from enum import StrEnum, Enum
import re

class TaskStatus(StrEnum):
    """Task status string constants"""
{self._generate_string_enum_values(status_grouping, "task_status")}

class ProcessingStatus(StrEnum):
    """Processing status string constants"""  
{self._generate_string_enum_values(status_grouping, "processing_status")}

class SystemStatus(StrEnum):
    """System status string constants"""
{self._generate_string_enum_values(status_grouping, "system_status")}

class MessageTemplates(StrEnum):
    """Message template constants"""
{self._generate_message_templates(status_grouping)}

class StatusCategories(Enum):
    """Status category groupings"""
    TASK = [status.value for status in TaskStatus]
    PROCESSING = [status.value for status in ProcessingStatus]  
    SYSTEM = [status.value for status in SystemStatus]

# Status validation
VALID_STATUSES: Final[Set[str]] = {{
    *[status.value for status in TaskStatus],
    *[status.value for status in ProcessingStatus],
    *[status.value for status in SystemStatus]
}}

def is_valid_status(status: str) -> bool:
    """Validate if status string is recognized"""
    return status in VALID_STATUSES

def normalize_status(status: str) -> Optional[str]:
    """Normalize status string to canonical form"""
    status_upper = status.upper().replace(' ', '_').replace('-', '_')
    
    # Try each enum type
    for enum_class in [TaskStatus, ProcessingStatus, SystemStatus]:
        try:
            return enum_class[status_upper].value
        except KeyError:
            continue
            
    return None

def get_status_category(status: str) -> Optional[str]:
    """Get category of status string"""
    if status in [s.value for s in TaskStatus]:
        return "TASK"
    elif status in [s.value for s in ProcessingStatus]:
        return "PROCESSING"  
    elif status in [s.value for s in SystemStatus]:
        return "SYSTEM"
    return None

# Status transitions
VALID_TASK_TRANSITIONS: Final[Dict[TaskStatus, Set[TaskStatus]]] = {{
    TaskStatus.PENDING: {{TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED}},
    TaskStatus.IN_PROGRESS: {{TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PAUSED}},
    TaskStatus.PAUSED: {{TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED}},
    TaskStatus.COMPLETED: set(),  # Terminal state
    TaskStatus.FAILED: {{TaskStatus.PENDING}},  # Can retry
    TaskStatus.CANCELLED: set()  # Terminal state
}}

def is_valid_transition(from_status: TaskStatus, to_status: TaskStatus) -> bool:
    """Check if status transition is valid"""
    return to_status in VALID_TASK_TRANSITIONS.get(from_status, set())

class StatusFormatter:
    """Utility class for status formatting and display"""
    
    @staticmethod
    def format_display(status: str) -> str:
        """Format status for user display"""
        return status.replace('_', ' ').title()
        
    @staticmethod
    def format_log(status: str) -> str:
        """Format status for logging"""
        return f"[{{status.upper()}}]"
        
    @staticmethod
    def format_api(status: str) -> str:
        """Format status for API responses"""
        return status.lower()

# Message formatting with templates
def format_status_message(template: MessageTemplates, **kwargs) -> str:
    """Format status message using template"""
    return template.value.format(**kwargs)
'''

        self.generated_constants["status_constants.py"] = status_constants_code
        
        outputs = {
            "status_constants_module": status_constants_code
        }
        
        await self._complete_task(task, outputs)
        
    async def _validate_literal_elimination(self, task: AgentTask):
        """Verify all 159 magic literals have been eliminated"""
        
        # Re-scan files to count remaining literals
        remaining_literals = []
        
        for file_path in self.task_management_files:
            literals = await self._extract_literals_from_file(file_path)
            remaining_literals.extend(literals)
            
        eliminated_count = len(self.found_literals) - len(remaining_literals)
        elimination_percentage = (eliminated_count / len(self.found_literals)) * 100 if self.found_literals else 100
        
        # Update final metric
        self.coordinator.memory_store.update_coupling_metric("MagicLiterals", len(remaining_literals))
        
        validation_report = {
            "original_literal_count": len(self.found_literals),
            "remaining_literal_count": len(remaining_literals),
            "eliminated_count": eliminated_count,
            "elimination_percentage": elimination_percentage,
            "target_achieved": len(remaining_literals) == 0,
            "constants_modules_created": len(self.generated_constants),
            "categories_processed": len(self.constants_groupings),
            "type_safety_level": "Full" if len(remaining_literals) == 0 else "Partial"
        }
        
        if len(remaining_literals) > 0:
            validation_report["remaining_literals_by_file"] = self._group_remaining_by_file(remaining_literals)
            
        outputs = {
            "elimination_validation_report": validation_report
        }
        
        await self._complete_task(task, outputs)
        
    async def _extract_literals_from_file(self, file_path: Path) -> List[MagicLiteral]:
        """Extract magic literals from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            literals = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    # Skip None, True, False
                    if node.value in (None, True, False):
                        continue
                        
                    literal = MagicLiteral(
                        value=node.value,
                        literal_type=type(node.value).__name__,
                        category=LiteralCategory.DEFAULT_VALUE,  # Will be determined later
                        locations=[f"{file_path}:{node.lineno}"],
                        usage_context=self._get_context_around_node(content, node)
                    )
                    literals.append(literal)
                    
            return literals
            
        except Exception as e:
            self.logger.error(f"Error extracting literals from {file_path}: {e}")
            return []
            
    def _determine_category(self, literal: MagicLiteral) -> LiteralCategory:
        """Determine the category of a magic literal"""
        value_str = str(literal.value).lower()
        context_lower = literal.usage_context.lower()
        
        # Timing-related patterns
        if any(keyword in context_lower for keyword in ['timeout', 'delay', 'sleep', 'wait', 'interval']):
            return LiteralCategory.TIMING
            
        # Calculation patterns  
        elif any(keyword in context_lower for keyword in ['calculate', 'compute', 'multiply', 'divide', 'threshold']):
            return LiteralCategory.CALCULATION
            
        # Status strings
        elif literal.literal_type == 'str' and any(keyword in value_str for keyword in ['pending', 'complete', 'failed', 'success', 'error']):
            return LiteralCategory.STATUS_STRING
            
        # Configuration keys
        elif literal.literal_type == 'str' and ('config' in context_lower or 'setting' in context_lower):
            return LiteralCategory.CONFIGURATION_KEY
            
        # Validation rules
        elif any(keyword in context_lower for keyword in ['validate', 'check', 'verify', 'rule']):
            return LiteralCategory.VALIDATION_RULE
            
        # Message templates
        elif literal.literal_type == 'str' and ('{' in str(literal.value) or '%' in str(literal.value)):
            return LiteralCategory.MESSAGE_TEMPLATE
            
        # Thresholds
        elif isinstance(literal.value, (int, float)) and any(keyword in context_lower for keyword in ['limit', 'max', 'min', 'threshold']):
            return LiteralCategory.THRESHOLD
            
        # Default fallback
        else:
            return LiteralCategory.DEFAULT_VALUE
            
    def _suggest_constant_name(self, literal: MagicLiteral) -> str:
        """Suggest a constant name for the literal"""
        context_words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', literal.usage_context)
        
        # Extract meaningful words from context
        meaningful_words = [word.upper() for word in context_words 
                           if len(word) > 2 and word.lower() not in ['the', 'and', 'or', 'if', 'is', 'in']]
        
        if meaningful_words:
            base_name = '_'.join(meaningful_words[:3])  # Limit to 3 words
        else:
            base_name = f"{literal.category.value.upper()}_VALUE"
            
        # Add type suffix if needed
        if literal.literal_type == 'str':
            if not base_name.endswith('_MSG') and not base_name.endswith('_STATUS'):
                base_name += '_STRING'
        elif literal.literal_type in ['int', 'float']:
            if literal.category == LiteralCategory.TIMING:
                base_name += '_MS' if isinstance(literal.value, int) and literal.value < 1000 else '_SECONDS'
                
        return base_name
        
    def _get_context_around_node(self, content: str, node: ast.AST) -> str:
        """Get context around an AST node"""
        lines = content.split('\n')
        if hasattr(node, 'lineno') and 1 <= node.lineno <= len(lines):
            line_idx = node.lineno - 1
            start = max(0, line_idx - 1)
            end = min(len(lines), line_idx + 2)
            return ' '.join(lines[start:end])
        return ""
        
    def _generate_int_enum_values(self, grouping: ConstantsGrouping) -> str:
        """Generate integer enum values"""
        enum_lines = []
        for const_name, literal in grouping.constants.items():
            if isinstance(literal.value, int):
                enum_lines.append(f"    {const_name} = {literal.value}")
        return '\n'.join(enum_lines) if enum_lines else "    DEFAULT_VALUE = 1"
        
    def _count_by_type(self, literals: List[MagicLiteral]) -> Dict[str, int]:
        """Count literals by type"""
        counts = {}
        for literal in literals:
            counts[literal.literal_type] = counts.get(literal.literal_type, 0) + 1
        return counts
        
    def _count_by_priority(self, literals: List[MagicLiteral]) -> Dict[int, int]:
        """Count literals by priority"""
        counts = {}
        for literal in literals:
            counts[literal.replacement_priority] = counts.get(literal.replacement_priority, 0) + 1
        return counts
        
    async def _find_task_management_files(self) -> List[Path]:
        """Find all task management related files"""
        project_root = Path.cwd()
        
        search_patterns = [
            "**/task_management/**/*.py",
            "**/tasks/**/*.py", 
            "**/management/**/*.py",
            "**/core/task*.py",
            "**/agents/task*.py"
        ]
        
        found_files = []
        for pattern in search_patterns:
            files = list(project_root.glob(pattern))
            found_files.extend(files)
            
        # Remove duplicates
        unique_files = list(set(found_files))
        
        self.logger.info(f"Found {len(unique_files)} task management files")
        return unique_files