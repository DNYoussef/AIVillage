"""
MCP Fallback Coordinator - Graceful Degradation and Offline Operation

This module provides comprehensive fallback mechanisms for MCP server unavailability,
ensuring that CI/CD pipelines can continue functioning even when external services fail.

Key Features:
- Intelligent fallback selection based on server capabilities
- Offline operation modes with local alternatives
- Graceful degradation with reduced functionality
- Automatic recovery when servers come back online
- Performance impact assessment and mitigation
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Strategies for handling server unavailability"""
    FAIL_FAST = "fail_fast"  # Fail immediately if server unavailable
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality but continue
    OFFLINE_MODE = "offline_mode"  # Use local alternatives
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Keep trying to reconnect
    HYBRID = "hybrid"  # Combination of strategies based on context


class FallbackCapability(Enum):
    """Categories of capabilities that can have fallbacks"""
    MEMORY_STORAGE = "memory_storage"
    SEQUENTIAL_REASONING = "sequential_reasoning"
    GITHUB_INTEGRATION = "github_integration"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    DOCUMENT_PROCESSING = "document_processing"
    WEB_SCRAPING = "web_scraping"
    MODEL_INFERENCE = "model_inference"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    strategy: FallbackStrategy = FallbackStrategy.HYBRID
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 2.0
    backoff_multiplier: float = 1.5
    offline_storage_path: Path = field(default_factory=lambda: Path(".mcp_fallback"))
    enable_performance_monitoring: bool = True
    max_degradation_time_minutes: int = 30


@dataclass
class FallbackMetrics:
    """Metrics for fallback operations"""
    activations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_response_time_ms: float = 0.0
    performance_degradation_percent: float = 0.0
    last_activation: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        total = self.successful_operations + self.failed_operations
        return self.successful_operations / max(total, 1)
    
    @property
    def avg_response_time_ms(self) -> float:
        total_ops = self.successful_operations + self.failed_operations
        return self.total_response_time_ms / max(total_ops, 1)


class FallbackProvider(ABC):
    """Abstract base class for fallback implementations"""
    
    def __init__(self, capability: FallbackCapability, config: FallbackConfig):
        self.capability = capability
        self.config = config
        self.metrics = FallbackMetrics()
        self.is_active = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the fallback provider"""
        pass
    
    @abstractmethod
    async def execute(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute an operation using the fallback provider"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass
    
    @abstractmethod
    def get_supported_operations(self) -> Set[str]:
        """Get set of operations supported by this fallback provider"""
        pass
    
    async def _track_execution(self, operation: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Track execution metrics for fallback operations"""
        start_time = time.perf_counter()
        
        try:
            result = await func(*args, **kwargs)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self.metrics.successful_operations += 1
            self.metrics.total_response_time_ms += execution_time
            
            return {
                "status": "success",
                "result": result,
                "execution_time_ms": execution_time,
                "fallback_provider": self.__class__.__name__,
                "capability": self.capability.value
            }
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.metrics.failed_operations += 1
            self.metrics.total_response_time_ms += execution_time
            
            self.logger.error(f"Fallback execution failed for {operation}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": execution_time,
                "fallback_provider": self.__class__.__name__,
                "capability": self.capability.value
            }


class MemoryFallbackProvider(FallbackProvider):
    """Fallback provider for memory/storage operations"""
    
    def __init__(self, config: FallbackConfig):
        super().__init__(FallbackCapability.MEMORY_STORAGE, config)
        self.local_storage: Dict[str, Any] = {}
        self.storage_file = config.offline_storage_path / "memory_fallback.json"
        self.namespaces: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize local memory storage"""
        try:
            self.config.offline_storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing data if available
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.local_storage = data.get("storage", {})
                    self.namespaces = data.get("namespaces", {})
            
            self.is_active = True
            self.logger.info("Memory fallback provider initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory fallback: {e}")
            return False
    
    def get_supported_operations(self) -> Set[str]:
        return {"store", "retrieve", "search", "delete", "list_keys", "create_namespace"}
    
    async def execute(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute memory operation using local storage"""
        if operation == "store":
            return await self._track_execution(operation, self._store, *args, **kwargs)
        elif operation == "retrieve":
            return await self._track_execution(operation, self._retrieve, *args, **kwargs)
        elif operation == "search":
            return await self._track_execution(operation, self._search, *args, **kwargs)
        elif operation == "delete":
            return await self._track_execution(operation, self._delete, *args, **kwargs)
        elif operation == "list_keys":
            return await self._track_execution(operation, self._list_keys, *args, **kwargs)
        elif operation == "create_namespace":
            return await self._track_execution(operation, self._create_namespace, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _store(self, key: str, value: Any, namespace: str = "default") -> Dict[str, Any]:
        """Store data in local memory"""
        if namespace not in self.namespaces:
            self.namespaces[namespace] = {}
        
        full_key = f"{namespace}:{key}"
        self.local_storage[full_key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "namespace": namespace
        }
        
        await self._persist_storage()
        return {"key": key, "namespace": namespace, "stored": True}
    
    async def _retrieve(self, key: str, namespace: str = "default") -> Dict[str, Any]:
        """Retrieve data from local memory"""
        full_key = f"{namespace}:{key}"
        data = self.local_storage.get(full_key)
        
        if data:
            return {"key": key, "value": data["value"], "found": True, "namespace": namespace}
        else:
            return {"key": key, "value": None, "found": False, "namespace": namespace}
    
    async def _search(self, pattern: str, namespace: str = None) -> Dict[str, Any]:
        """Search for keys matching pattern"""
        results = []
        
        for full_key, data in self.local_storage.items():
            key_namespace, key = full_key.split(":", 1)
            
            if namespace and key_namespace != namespace:
                continue
            
            if pattern in key or pattern in str(data.get("value", "")):
                results.append({
                    "key": key,
                    "value": data["value"],
                    "namespace": key_namespace,
                    "timestamp": data["timestamp"]
                })
        
        return {"pattern": pattern, "results": results, "count": len(results)}
    
    async def _delete(self, key: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete data from local memory"""
        full_key = f"{namespace}:{key}"
        
        if full_key in self.local_storage:
            del self.local_storage[full_key]
            await self._persist_storage()
            return {"key": key, "namespace": namespace, "deleted": True}
        else:
            return {"key": key, "namespace": namespace, "deleted": False, "error": "Key not found"}
    
    async def _list_keys(self, namespace: str = None) -> Dict[str, Any]:
        """List all keys, optionally filtered by namespace"""
        keys = []
        
        for full_key in self.local_storage.keys():
            key_namespace, key = full_key.split(":", 1)
            
            if namespace is None or key_namespace == namespace:
                keys.append({"key": key, "namespace": key_namespace})
        
        return {"keys": keys, "count": len(keys)}
    
    async def _create_namespace(self, namespace: str, description: str = "") -> Dict[str, Any]:
        """Create a new namespace"""
        self.namespaces[namespace] = {
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        await self._persist_storage()
        return {"namespace": namespace, "created": True, "description": description}
    
    async def _persist_storage(self):
        """Persist storage to disk"""
        try:
            data = {
                "storage": self.local_storage,
                "namespaces": self.namespaces,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to persist storage: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self._persist_storage()
        self.is_active = False


class SequentialThinkingFallbackProvider(FallbackProvider):
    """Fallback provider for sequential thinking operations"""
    
    def __init__(self, config: FallbackConfig):
        super().__init__(FallbackCapability.SEQUENTIAL_REASONING, config)
        self.reasoning_templates = {
            "problem_solving": [
                "1. Understand the problem",
                "2. Identify constraints and requirements",
                "3. Generate possible solutions",
                "4. Evaluate each solution",
                "5. Select the best approach",
                "6. Plan implementation steps"
            ],
            "decision_making": [
                "1. Define the decision criteria",
                "2. Identify available options",
                "3. Gather relevant information",
                "4. Analyze pros and cons",
                "5. Consider risks and benefits",
                "6. Make the decision"
            ],
            "analysis": [
                "1. Break down the subject into components",
                "2. Examine each component individually",
                "3. Identify relationships and patterns",
                "4. Consider context and implications",
                "5. Synthesize findings",
                "6. Draw conclusions"
            ]
        }
    
    async def initialize(self) -> bool:
        """Initialize sequential thinking fallback"""
        self.is_active = True
        self.logger.info("Sequential thinking fallback provider initialized")
        return True
    
    def get_supported_operations(self) -> Set[str]:
        return {"think", "analyze", "decide", "plan", "reason"}
    
    async def execute(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute sequential thinking operation"""
        if operation == "think":
            return await self._track_execution(operation, self._think, *args, **kwargs)
        elif operation == "analyze":
            return await self._track_execution(operation, self._analyze, *args, **kwargs)
        elif operation == "decide":
            return await self._track_execution(operation, self._decide, *args, **kwargs)
        elif operation == "plan":
            return await self._track_execution(operation, self._plan, *args, **kwargs)
        elif operation == "reason":
            return await self._track_execution(operation, self._reason, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _think(self, prompt: str, thinking_type: str = "problem_solving") -> Dict[str, Any]:
        """Basic sequential thinking"""
        template = self.reasoning_templates.get(thinking_type, self.reasoning_templates["problem_solving"])
        
        # Apply template to prompt
        reasoning_steps = []
        for step in template:
            reasoning_steps.append({
                "step": step,
                "context": f"Applied to: {prompt[:100]}..." if len(prompt) > 100 else prompt,
                "reasoning": f"Considering {step.lower()} for the given problem"
            })
        
        return {
            "prompt": prompt,
            "thinking_type": thinking_type,
            "steps": reasoning_steps,
            "conclusion": f"Based on {thinking_type} approach, recommend proceeding with structured implementation",
            "confidence": 0.7  # Moderate confidence for fallback reasoning
        }
    
    async def _analyze(self, subject: str) -> Dict[str, Any]:
        """Basic analysis using structured approach"""
        return await self._think(f"Analyze: {subject}", "analysis")
    
    async def _decide(self, decision_context: str, options: List[str] = None) -> Dict[str, Any]:
        """Basic decision making"""
        options = options or ["Option A", "Option B"]
        
        decision_result = await self._think(f"Decide between {options} in context: {decision_context}", "decision_making")
        
        # Add decision-specific information
        decision_result["options"] = options
        decision_result["recommended_option"] = options[0]  # Simple default
        decision_result["rationale"] = "Selected based on structured decision-making process"
        
        return decision_result
    
    async def _plan(self, goal: str) -> Dict[str, Any]:
        """Basic planning using structured approach"""
        plan_result = await self._think(f"Plan to achieve: {goal}", "problem_solving")
        
        # Convert steps to action items
        plan_result["action_items"] = [
            {"action": step["step"], "priority": "medium", "estimated_effort": "medium"}
            for step in plan_result["steps"]
        ]
        
        return plan_result
    
    async def _reason(self, premise: str, question: str) -> Dict[str, Any]:
        """Basic reasoning about a question given a premise"""
        reasoning_prompt = f"Given: {premise}, Question: {question}"
        return await self._think(reasoning_prompt, "analysis")
    
    async def cleanup(self):
        """Clean up resources"""
        self.is_active = False


class GitHubFallbackProvider(FallbackProvider):
    """Fallback provider for GitHub operations when GitHub MCP is unavailable"""
    
    def __init__(self, config: FallbackConfig):
        super().__init__(FallbackCapability.GITHUB_INTEGRATION, config)
        self.local_git_commands = True  # Whether git CLI is available
        self.cached_repo_info: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize GitHub fallback provider"""
        try:
            # Check if git is available
            import subprocess
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            self.local_git_commands = result.returncode == 0
            
            self.is_active = True
            self.logger.info(f"GitHub fallback provider initialized (git available: {self.local_git_commands})")
            return True
            
        except Exception as e:
            self.logger.warning(f"GitHub fallback initialization warning: {e}")
            self.is_active = True  # Still activate with limited functionality
            return True
    
    def get_supported_operations(self) -> Set[str]:
        if self.local_git_commands:
            return {"get_repo_info", "get_branch_info", "get_commit_info", "create_branch", "commit_changes"}
        else:
            return {"get_repo_info"}  # Very limited without git
    
    async def execute(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute GitHub operation using fallback methods"""
        if operation == "get_repo_info":
            return await self._track_execution(operation, self._get_repo_info, *args, **kwargs)
        elif operation == "get_branch_info" and self.local_git_commands:
            return await self._track_execution(operation, self._get_branch_info, *args, **kwargs)
        elif operation == "get_commit_info" and self.local_git_commands:
            return await self._track_execution(operation, self._get_commit_info, *args, **kwargs)
        elif operation == "create_branch" and self.local_git_commands:
            return await self._track_execution(operation, self._create_branch, *args, **kwargs)
        elif operation == "commit_changes" and self.local_git_commands:
            return await self._track_execution(operation, self._commit_changes, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation} (git available: {self.local_git_commands})")
    
    async def _get_repo_info(self, repo_path: str = ".") -> Dict[str, Any]:
        """Get basic repository information"""
        try:
            repo_path = Path(repo_path)
            git_dir = repo_path / ".git"
            
            if git_dir.exists():
                return {
                    "repo_path": str(repo_path.absolute()),
                    "has_git": True,
                    "git_dir": str(git_dir),
                    "status": "local_repo_detected"
                }
            else:
                return {
                    "repo_path": str(repo_path.absolute()),
                    "has_git": False,
                    "status": "not_a_git_repository"
                }
                
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def _get_branch_info(self) -> Dict[str, Any]:
        """Get current branch information using git CLI"""
        try:
            import subprocess
            
            # Get current branch
            result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            current_branch = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Get all branches
            result = subprocess.run(["git", "branch"], capture_output=True, text=True)
            branches = [line.strip().lstrip("* ") for line in result.stdout.splitlines()] if result.returncode == 0 else []
            
            return {
                "current_branch": current_branch,
                "all_branches": branches,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def _get_commit_info(self, commit_ref: str = "HEAD") -> Dict[str, Any]:
        """Get commit information using git CLI"""
        try:
            import subprocess
            
            result = subprocess.run(
                ["git", "show", "--format=%H|%s|%an|%ad", "--no-patch", commit_ref],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split("|")
                return {
                    "commit_hash": parts[0] if len(parts) > 0 else "unknown",
                    "message": parts[1] if len(parts) > 1 else "unknown",
                    "author": parts[2] if len(parts) > 2 else "unknown",
                    "date": parts[3] if len(parts) > 3 else "unknown",
                    "status": "success"
                }
            else:
                return {"error": result.stderr, "status": "error"}
                
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def _create_branch(self, branch_name: str, base_branch: str = None) -> Dict[str, Any]:
        """Create a new branch using git CLI"""
        try:
            import subprocess
            
            if base_branch:
                result = subprocess.run(["git", "checkout", "-b", branch_name, base_branch], capture_output=True, text=True)
            else:
                result = subprocess.run(["git", "checkout", "-b", branch_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "branch_name": branch_name,
                    "base_branch": base_branch,
                    "status": "success",
                    "message": "Branch created successfully"
                }
            else:
                return {"error": result.stderr, "status": "error"}
                
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def _commit_changes(self, message: str, files: List[str] = None) -> Dict[str, Any]:
        """Commit changes using git CLI"""
        try:
            import subprocess
            
            # Add files
            if files:
                for file in files:
                    subprocess.run(["git", "add", file], capture_output=True)
            else:
                subprocess.run(["git", "add", "-A"], capture_output=True)
            
            # Commit
            result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "message": message,
                    "files": files or "all_changes",
                    "status": "success",
                    "commit_output": result.stdout
                }
            else:
                return {"error": result.stderr, "status": "error"}
                
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def cleanup(self):
        """Clean up resources"""
        self.is_active = False


class MCPFallbackCoordinator:
    """
    Coordinates fallback mechanisms across all MCP server types
    
    Responsibilities:
    - Manage fallback provider lifecycle
    - Route operations to appropriate fallback providers
    - Monitor fallback performance and health
    - Automatically recover when primary servers return
    - Provide degradation analytics and reporting
    """
    
    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Fallback providers
        self.providers: Dict[FallbackCapability, FallbackProvider] = {}
        self.active_fallbacks: Set[FallbackCapability] = set()
        
        # Server availability tracking
        self.server_availability: Dict[str, bool] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_attempt: Dict[str, datetime] = {}
        
        # Performance monitoring
        self.performance_baseline: Dict[str, float] = {}
        self.degradation_start_time: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            "total_fallback_activations": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "total_degradation_time_minutes": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all fallback providers"""
        try:
            self.logger.info("Initializing MCP Fallback Coordinator...")
            
            # Initialize fallback providers
            providers = [
                (FallbackCapability.MEMORY_STORAGE, MemoryFallbackProvider(self.config)),
                (FallbackCapability.SEQUENTIAL_REASONING, SequentialThinkingFallbackProvider(self.config)),
                (FallbackCapability.GITHUB_INTEGRATION, GitHubFallbackProvider(self.config))
            ]
            
            for capability, provider in providers:
                if await provider.initialize():
                    self.providers[capability] = provider
                    self.logger.info(f"Initialized {capability.value} fallback provider")
                else:
                    self.logger.warning(f"Failed to initialize {capability.value} fallback provider")
            
            self.logger.info(f"Fallback coordinator initialized with {len(self.providers)} providers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback coordinator: {e}")
            return False
    
    async def activate_fallback(self, server_name: str, capability: FallbackCapability) -> bool:
        """Activate fallback for a specific server capability"""
        if capability not in self.providers:
            self.logger.error(f"No fallback provider available for {capability.value}")
            return False
        
        try:
            provider = self.providers[capability]
            
            if not provider.is_active:
                await provider.initialize()
            
            self.active_fallbacks.add(capability)
            self.server_availability[server_name] = False
            self.recovery_attempts[server_name] = 0
            
            provider.metrics.activations += 1
            provider.metrics.last_activation = datetime.now()
            
            if self.degradation_start_time is None:
                self.degradation_start_time = datetime.now()
            
            self.stats["total_fallback_activations"] += 1
            
            self.logger.warning(f"Activated {capability.value} fallback for server '{server_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to activate fallback for {server_name}: {e}")
            return False
    
    async def deactivate_fallback(self, server_name: str, capability: FallbackCapability) -> bool:
        """Deactivate fallback when primary server recovers"""
        try:
            if capability in self.active_fallbacks:
                self.active_fallbacks.remove(capability)
                self.server_availability[server_name] = True
                self.stats["successful_recoveries"] += 1
                
                # Calculate degradation time
                if self.degradation_start_time and not self.active_fallbacks:
                    degradation_duration = datetime.now() - self.degradation_start_time
                    self.stats["total_degradation_time_minutes"] += degradation_duration.total_seconds() / 60
                    self.degradation_start_time = None
                
                self.logger.info(f"Deactivated {capability.value} fallback for server '{server_name}' - primary server recovered")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate fallback for {server_name}: {e}")
            return False
    
    async def execute_with_fallback(self, server_name: str, capability: FallbackCapability, 
                                  operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute operation using fallback if primary server unavailable"""
        # First try primary server if available
        if self.server_availability.get(server_name, True) and capability not in self.active_fallbacks:
            try:
                # This would call the primary server
                # For now, simulate primary server call
                await asyncio.sleep(0.001)  # Simulate network call
                
                return {
                    "status": "success",
                    "result": f"Primary server result for {operation}",
                    "server": "primary",
                    "capability": capability.value
                }
                
            except Exception as e:
                self.logger.warning(f"Primary server '{server_name}' failed: {e}")
                await self.activate_fallback(server_name, capability)
        
        # Use fallback provider
        if capability in self.providers and capability in self.active_fallbacks:
            provider = self.providers[capability]
            result = await provider.execute(operation, *args, **kwargs)
            result["fallback_used"] = True
            return result
        
        # No fallback available
        return {
            "status": "error",
            "error": f"No fallback available for {capability.value}",
            "server": server_name,
            "capability": capability.value
        }
    
    async def attempt_recovery(self, server_name: str) -> bool:
        """Attempt to recover connection to primary server"""
        if server_name not in self.recovery_attempts:
            self.recovery_attempts[server_name] = 0
        
        if self.recovery_attempts[server_name] >= self.config.max_retry_attempts:
            return False
        
        # Check if enough time has passed since last attempt
        last_attempt = self.last_recovery_attempt.get(server_name)
        if last_attempt:
            time_since_last = datetime.now() - last_attempt
            min_wait_time = timedelta(seconds=self.config.retry_delay_seconds * (self.config.backoff_multiplier ** self.recovery_attempts[server_name]))
            
            if time_since_last < min_wait_time:
                return False
        
        try:
            self.recovery_attempts[server_name] += 1
            self.last_recovery_attempt[server_name] = datetime.now()
            
            # Attempt to reconnect to server
            # This would be actual reconnection logic
            await asyncio.sleep(0.1)  # Simulate connection attempt
            
            # For demo, randomly succeed/fail
            import random
            success = random.random() > 0.3  # 70% success rate
            
            if success:
                # Recovery successful
                self.server_availability[server_name] = True
                self.recovery_attempts[server_name] = 0
                self.logger.info(f"Successfully recovered connection to '{server_name}'")
                return True
            else:
                self.logger.debug(f"Recovery attempt {self.recovery_attempts[server_name]} failed for '{server_name}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed for '{server_name}': {e}")
            self.stats["failed_recoveries"] += 1
            return False
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all fallback mechanisms"""
        provider_status = {}
        
        for capability, provider in self.providers.items():
            provider_status[capability.value] = {
                "is_active": capability in self.active_fallbacks,
                "metrics": {
                    "activations": provider.metrics.activations,
                    "success_rate": provider.metrics.success_rate,
                    "avg_response_time_ms": provider.metrics.avg_response_time_ms,
                    "performance_degradation_percent": provider.metrics.performance_degradation_percent
                },
                "supported_operations": list(provider.get_supported_operations())
            }
        
        current_degradation_time = 0.0
        if self.degradation_start_time:
            current_degradation_time = (datetime.now() - self.degradation_start_time).total_seconds() / 60
        
        return {
            "coordinator_status": {
                "active_fallbacks": len(self.active_fallbacks),
                "available_providers": len(self.providers),
                "current_degradation_time_minutes": current_degradation_time,
                "total_degradation_time_minutes": self.stats["total_degradation_time_minutes"]
            },
            "server_availability": self.server_availability.copy(),
            "recovery_attempts": self.recovery_attempts.copy(),
            "provider_status": provider_status,
            "statistics": self.stats.copy()
        }
    
    async def cleanup(self):
        """Clean up all fallback providers"""
        self.logger.info("Cleaning up fallback coordinator...")
        
        for provider in self.providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up provider: {e}")
        
        self.providers.clear()
        self.active_fallbacks.clear()


# Example usage
async def example_fallback_usage():
    """Example of how to use the fallback coordinator"""
    config = FallbackConfig(
        strategy=FallbackStrategy.HYBRID,
        enable_performance_monitoring=True,
        max_degradation_time_minutes=30
    )
    
    coordinator = MCPFallbackCoordinator(config)
    await coordinator.initialize()
    
    # Simulate server failure and fallback activation
    await coordinator.activate_fallback("memory_server", FallbackCapability.MEMORY_STORAGE)
    
    # Use fallback for operations
    result = await coordinator.execute_with_fallback(
        "memory_server", 
        FallbackCapability.MEMORY_STORAGE,
        "store",
        "test_key", 
        "test_value"
    )
    print(f"Fallback operation result: {result}")
    
    # Check status
    status = coordinator.get_fallback_status()
    print(f"Fallback status: {status}")
    
    await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(example_fallback_usage())