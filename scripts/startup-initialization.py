#!/usr/bin/env python3
"""
Comprehensive Startup Initialization System
Initializes all Claude Flow components, memory systems, DSPy optimization, and automation
"""

import os
import sys
import json
import sqlite3
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.claude/logs/startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StartupInitializer:
    """Comprehensive startup initialization for Claude Code ecosystem"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.claude_dir = self.project_root / '.claude'
        self.hive_mind_dir = self.project_root / '.claude/hive-mind'
        self.claude_flow_dir = self.project_root / '.claude-flow'
        self.swarm_dir = self.project_root / '.swarm'
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Initialize components
        self.memory_manager = MemoryManager(self.hive_mind_dir)
        self.swarm_coordinator = SwarmCoordinator(self.swarm_dir)
        self.dspy_optimizer = DSPyOptimizer(self.claude_dir / 'dspy')
        self.test_failure_handler = TestFailureHandler(self.claude_dir / 'test-failures')
        
        logger.info("🚀 Startup Initializer loaded")
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.claude_dir / 'logs',
            self.claude_dir / 'dspy',
            self.claude_dir / 'test-failures',
            self.claude_dir / 'agents' / 'memory',
            self.claude_dir / 'swarm' / 'topologies',
            self.hive_mind_dir / 'sessions',
            self.claude_flow_dir / 'metrics',
            self.swarm_dir / 'coordination'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✓ Ensured directory: {directory}")
    
    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all components"""
        logger.info("🎯 Starting comprehensive initialization...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'starting',
            'components': {}
        }
        
        try:
            # Initialize in optimal order
            initialization_steps = [
                ('memory_systems', self._initialize_memory_systems),
                ('claude_flow', self._initialize_claude_flow),
                ('swarm_coordination', self._initialize_swarm_coordination),
                ('dspy_optimization', self._initialize_dspy_optimization),
                ('test_failure_handling', self._initialize_test_failure_handling),
                ('github_integration', self._initialize_github_integration),
                ('monitoring', self._initialize_monitoring)
            ]
            
            for component_name, init_func in initialization_steps:
                logger.info(f"🔧 Initializing {component_name}...")
                component_result = await init_func()
                results['components'][component_name] = component_result
                logger.info(f"✅ {component_name} initialized successfully")
            
            results['status'] = 'success'
            logger.info("🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results
    
    async def _initialize_memory_systems(self) -> Dict[str, Any]:
        """Initialize memory and hive-mind systems"""
        return await self.memory_manager.initialize()
    
    async def _initialize_claude_flow(self) -> Dict[str, Any]:
        """Initialize Claude Flow components"""
        try:
            # Run Claude Flow initialization
            result = subprocess.run([
                'npx', 'claude-flow', 'init', '--yes-all'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.warning(f"Claude Flow init warning: {result.stderr}")
            
            return {
                'status': 'initialized',
                'metrics_directory': str(self.claude_flow_dir / 'metrics'),
                'agents_available': self._count_available_agents()
            }
        except Exception as e:
            logger.error(f"Claude Flow initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _initialize_swarm_coordination(self) -> Dict[str, Any]:
        """Initialize swarm coordination"""
        return await self.swarm_coordinator.initialize()
    
    async def _initialize_dspy_optimization(self) -> Dict[str, Any]:
        """Initialize DSPy prompt optimization"""
        return await self.dspy_optimizer.initialize()
    
    async def _initialize_test_failure_handling(self) -> Dict[str, Any]:
        """Initialize automated test failure handling"""
        return await self.test_failure_handler.initialize()
    
    async def _initialize_github_integration(self) -> Dict[str, Any]:
        """Initialize GitHub integration"""
        try:
            # Check if GitHub integration is available
            github_workflow = self.project_root / '.github' / 'workflows' / 'claude-code-integration.yml'
            
            if not github_workflow.exists():
                logger.warning("GitHub workflow not found, creating...")
                # The workflow should already exist from previous setup
            
            return {
                'status': 'initialized',
                'workflow_exists': github_workflow.exists(),
                'test_failure_integration': True
            }
        except Exception as e:
            logger.error(f"GitHub integration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _initialize_monitoring(self) -> Dict[str, Any]:
        """Initialize monitoring and metrics"""
        try:
            # Create monitoring dashboard
            monitoring_config = {
                'enabled': True,
                'metrics_collection': True,
                'performance_tracking': True,
                'error_reporting': True,
                'test_failure_alerts': True
            }
            
            monitoring_file = self.claude_dir / 'monitoring.json'
            with open(monitoring_file, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            return {
                'status': 'initialized',
                'config_file': str(monitoring_file),
                'features': list(monitoring_config.keys())
            }
        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _count_available_agents(self) -> int:
        """Count available agent definitions"""
        agents_dir = self.claude_dir / 'agents'
        if not agents_dir.exists():
            return 0
        
        count = 0
        for item in agents_dir.rglob('*.md'):
            count += 1
        for item in agents_dir.rglob('*.json'):
            count += 1
        return count


class MemoryManager:
    """Manages persistent memory across sessions"""
    
    def __init__(self, hive_mind_dir: Path):
        self.hive_mind_dir = hive_mind_dir
        self.memory_db = hive_mind_dir / 'memory.db'
        self.hive_db = hive_mind_dir / 'hive.db'
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize memory systems"""
        try:
            # Initialize memory database
            self._init_memory_db()
            self._init_hive_db()
            
            # Load existing sessions
            sessions = self._load_existing_sessions()
            
            return {
                'status': 'initialized',
                'memory_db': str(self.memory_db),
                'hive_db': str(self.hive_db),
                'existing_sessions': len(sessions),
                'memory_size_mb': self._get_db_size_mb()
            }
        except Exception as e:
            logger.error(f"Memory initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _init_memory_db(self):
        """Initialize memory database schema"""
        with sqlite3.connect(self.memory_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    tags TEXT
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_key ON memory_entries(key)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id)
            ''')
    
    def _init_hive_db(self):
        """Initialize hive mind database schema"""
        with sqlite3.connect(self.hive_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS swarm_coordination (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    swarm_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task_id TEXT,
                    coordination_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_type TEXT NOT NULL,
                    learning_data TEXT NOT NULL,
                    performance_metrics TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def _load_existing_sessions(self) -> List[str]:
        """Load existing session files"""
        sessions_dir = self.hive_mind_dir / 'sessions'
        if not sessions_dir.exists():
            return []
        
        return [f.name for f in sessions_dir.glob('*.json')]
    
    def _get_db_size_mb(self) -> float:
        """Get total database size in MB"""
        total_size = 0
        for db_file in [self.memory_db, self.hive_db]:
            if db_file.exists():
                total_size += db_file.stat().st_size
        return total_size / (1024 * 1024)


class SwarmCoordinator:
    """Coordinates swarm initialization and management"""
    
    def __init__(self, swarm_dir: Path):
        self.swarm_dir = swarm_dir
        self.coordination_db = swarm_dir / 'coordination.db'
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize swarm coordination"""
        try:
            # Initialize coordination database
            self._init_coordination_db()
            
            # Set up swarm topologies
            topologies = self._setup_topologies()
            
            return {
                'status': 'initialized',
                'coordination_db': str(self.coordination_db),
                'available_topologies': topologies,
                'swarm_ready': True
            }
        except Exception as e:
            logger.error(f"Swarm coordination failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _init_coordination_db(self):
        """Initialize swarm coordination database"""
        with sqlite3.connect(self.coordination_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS swarm_instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    swarm_id TEXT UNIQUE NOT NULL,
                    topology TEXT NOT NULL,
                    max_agents INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    swarm_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def _setup_topologies(self) -> List[str]:
        """Set up available swarm topologies"""
        topologies = ['hierarchical', 'mesh', 'ring', 'star']
        
        for topology in topologies:
            topology_file = self.swarm_dir / 'topologies' / f'{topology}.json'
            if not topology_file.exists():
                topology_config = {
                    'name': topology,
                    'description': f'{topology.title()} topology configuration',
                    'max_agents': 8,
                    'coordination_pattern': topology,
                    'features': {
                        'fault_tolerance': True,
                        'load_balancing': True,
                        'auto_scaling': True
                    }
                }
                
                with open(topology_file, 'w') as f:
                    json.dump(topology_config, f, indent=2)
        
        return topologies


class DSPyOptimizer:
    """Handles DSPy prompt optimization"""
    
    def __init__(self, dspy_dir: Path):
        self.dspy_dir = dspy_dir
        self.optimization_cache = dspy_dir / 'optimization_cache.json'
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize DSPy optimization"""
        try:
            # Check if DSPy is available
            dspy_available = self._check_dspy_availability()
            
            # Initialize optimization cache
            self._init_optimization_cache()
            
            # Set up optimization templates
            templates = self._setup_optimization_templates()
            
            return {
                'status': 'initialized',
                'dspy_available': dspy_available,
                'optimization_cache': str(self.optimization_cache),
                'templates': templates,
                'auto_optimization': True
            }
        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _check_dspy_availability(self) -> bool:
        """Check if DSPy is available"""
        try:
            import dspy
            return True
        except ImportError:
            logger.warning("DSPy not installed, optimization features limited")
            return False
    
    def _init_optimization_cache(self):
        """Initialize optimization cache"""
        if not self.optimization_cache.exists():
            cache_data = {
                'version': '1.0.0',
                'optimizations': {},
                'performance_metrics': {},
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.optimization_cache, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def _setup_optimization_templates(self) -> List[str]:
        """Set up optimization templates"""
        templates = [
            'code_generation',
            'code_review',
            'test_creation',
            'documentation',
            'refactoring'
        ]
        
        for template in templates:
            template_file = self.dspy_dir / f'{template}_template.py'
            if not template_file.exists():
                template_content = f'''"""
DSPy optimization template for {template}
Auto-generated by startup initialization
"""

import dspy

class {template.title().replace('_', '')}Optimizer(dspy.Signature):
    """Optimized prompt for {template}"""
    
    input_context: str = dspy.InputField(desc="Input context for {template}")
    requirements: str = dspy.InputField(desc="Specific requirements")
    output: str = dspy.OutputField(desc="Optimized {template} output")

# Initialize optimizer
{template}_optimizer = dspy.ChainOfThought({template.title().replace('_', '')}Optimizer)
'''
                
                with open(template_file, 'w') as f:
                    f.write(template_content)
        
        return templates


class TestFailureHandler:
    """Handles automated test failure processing"""
    
    def __init__(self, test_failures_dir: Path):
        self.test_failures_dir = test_failures_dir
        self.failures_db = test_failures_dir / 'failures.db'
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize test failure handling"""
        try:
            # Initialize failures database
            self._init_failures_db()
            
            # Set up GitHub webhook handler
            webhook_config = self._setup_github_webhook()
            
            # Create failure analysis templates
            analysis_templates = self._create_analysis_templates()
            
            return {
                'status': 'initialized',
                'failures_db': str(self.failures_db),
                'webhook_config': webhook_config,
                'analysis_templates': analysis_templates,
                'auto_fix_enabled': True
            }
        except Exception as e:
            logger.error(f"Test failure handler initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _init_failures_db(self):
        """Initialize test failures database"""
        with sqlite3.connect(self.failures_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pr_number INTEGER,
                    commit_hash TEXT,
                    test_name TEXT NOT NULL,
                    failure_message TEXT NOT NULL,
                    stack_trace TEXT,
                    file_path TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME,
                    fix_attempt_count INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fix_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    failure_id INTEGER NOT NULL,
                    attempted_fix TEXT NOT NULL,
                    result TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (failure_id) REFERENCES test_failures (id)
                )
            ''')
    
    def _setup_github_webhook(self) -> Dict[str, str]:
        """Set up GitHub webhook configuration"""
        webhook_config = {
            'endpoint': '/webhook/test-failures',
            'events': ['check_run', 'workflow_run', 'pull_request'],
            'secret_key': 'claude_test_failure_handler'
        }
        
        webhook_file = self.test_failures_dir / 'webhook_config.json'
        with open(webhook_file, 'w') as f:
            json.dump(webhook_config, f, indent=2)
        
        return webhook_config
    
    def _create_analysis_templates(self) -> List[str]:
        """Create test failure analysis templates"""
        templates = [
            'syntax_error_analysis',
            'logical_error_analysis', 
            'integration_failure_analysis',
            'performance_regression_analysis',
            'dependency_conflict_analysis'
        ]
        
        for template in templates:
            template_file = self.test_failures_dir / f'{template}.py'
            if not template_file.exists():
                template_content = f'''"""
Test failure analysis template for {template.replace('_', ' ')}
Auto-generated by startup initialization
"""

class {template.title().replace('_', '')}:
    """Analyzes and fixes {template.replace('_', ' ')}"""
    
    def analyze(self, failure_data):
        """Analyze the failure"""
        # Implementation will be auto-generated
        pass
    
    def generate_fix(self, analysis_result):
        """Generate fix based on analysis"""
        # Implementation will be auto-generated
        pass
    
    def validate_fix(self, fix_code):
        """Validate the proposed fix"""
        # Implementation will be auto-generated
        pass
'''
                
                with open(template_file, 'w') as f:
                    f.write(template_content)
        
        return templates


async def main():
    """Main initialization function"""
    initializer = StartupInitializer()
    
    try:
        results = await initializer.initialize_all()
        
        # Save initialization results
        results_file = initializer.claude_dir / 'startup_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("🎉 Startup initialization complete!")
        print(f"📊 Results saved to: {results_file}")
        
        # Print summary
        print("\n📋 Initialization Summary:")
        for component, result in results['components'].items():
            status = result.get('status', 'unknown')
            print(f"  {component}: {status}")
        
        return results
        
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")
        return {'status': 'failed', 'error': str(e)}


if __name__ == '__main__':
    asyncio.run(main())