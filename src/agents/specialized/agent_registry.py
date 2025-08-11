"""
Specialized Agent Registry - Central management for all specialized sub-agents
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass

from src.production.rag.rag_system.core.agent_interface import AgentInterface

# Import all specialized agents
from .data_science_agent import DataScienceAgent
from .devops_agent import DevOpsAgent
from .financial_agent import FinancialAgent
from .creative_agent import CreativeAgent
from .social_agent import SocialAgent
from .translator_agent import TranslatorAgent
from .architect_agent import ArchitectAgent
from .tester_agent import TesterAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_type: str
    capability_name: str
    description: str
    input_types: List[str]
    output_types: List[str]


class SpecializedAgentRegistry:
    """
    Central registry for managing all specialized sub-agents.
    Provides discovery, instantiation, and coordination services.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentInterface] = {}
        self.agent_classes: Dict[str, Type[AgentInterface]] = {
            'data_science': DataScienceAgent,
            'devops': DevOpsAgent,
            'financial': FinancialAgent,
            'creative': CreativeAgent,
            'social': SocialAgent,
            'translator': TranslatorAgent,
            'architect': ArchitectAgent,
            'tester': TesterAgent
        }
        self.capabilities: Dict[str, List[AgentCapability]] = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the registry and all agents"""
        try:
            logger.info("Initializing Specialized Agent Registry...")
            
            # Register all agent capabilities
            await self._register_capabilities()
            
            # Initialize core agents
            core_agents = ['data_science', 'devops', 'tester']
            for agent_type in core_agents:
                await self.get_or_create_agent(agent_type)
            
            self.initialized = True
            logger.info(f"Registry initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {e}")
            self.initialized = False
    
    async def _register_capabilities(self):
        """Register capabilities for all agent types"""
        self.capabilities = {
            'data_science': [
                AgentCapability('data_science', 'statistical_analysis', 'Perform statistical analysis on datasets', ['csv', 'json'], ['analysis_report']),
                AgentCapability('data_science', 'ml_model_training', 'Train machine learning models', ['dataset'], ['trained_model', 'metrics']),
                AgentCapability('data_science', 'anomaly_detection', 'Detect anomalies in data', ['timeseries'], ['anomaly_report']),
                AgentCapability('data_science', 'time_series_analysis', 'Analyze time series data', ['timeseries'], ['forecast', 'trends'])
            ],
            'devops': [
                AgentCapability('devops', 'deployment_automation', 'Automate service deployments', ['deployment_config'], ['deployment_result']),
                AgentCapability('devops', 'infrastructure_management', 'Manage cloud infrastructure', ['infra_config'], ['provisioned_resources']),
                AgentCapability('devops', 'ci_cd_pipeline', 'Manage CI/CD pipelines', ['pipeline_config'], ['pipeline_status']),
                AgentCapability('devops', 'monitoring_setup', 'Setup monitoring and alerting', ['monitoring_config'], ['monitoring_dashboard'])
            ],
            'financial': [
                AgentCapability('financial', 'portfolio_optimization', 'Optimize investment portfolios', ['assets', 'constraints'], ['optimized_portfolio']),
                AgentCapability('financial', 'risk_analysis', 'Analyze financial risk', ['portfolio_data'], ['risk_metrics']),
                AgentCapability('financial', 'market_forecasting', 'Forecast market trends', ['market_data'], ['forecast', 'trends']),
                AgentCapability('financial', 'credit_scoring', 'Assess credit risk', ['borrower_data'], ['credit_score', 'risk_assessment'])
            ],
            'creative': [
                AgentCapability('creative', 'story_generation', 'Generate creative stories', ['theme', 'style'], ['story_outline']),
                AgentCapability('creative', 'visual_design', 'Create visual design concepts', ['brief'], ['design_concept']),
                AgentCapability('creative', 'music_composition', 'Compose music concepts', ['theme', 'style'], ['composition']),
                AgentCapability('creative', 'character_development', 'Develop fictional characters', ['character_brief'], ['character_profile'])
            ],
            'social': [
                AgentCapability('social', 'community_moderation', 'Moderate online communities', ['content'], ['moderation_action']),
                AgentCapability('social', 'conflict_resolution', 'Resolve conflicts between users', ['conflict_data'], ['resolution_plan']),
                AgentCapability('social', 'engagement_strategy', 'Develop community engagement strategies', ['community_data'], ['engagement_plan']),
                AgentCapability('social', 'sentiment_monitoring', 'Monitor community sentiment', ['interactions'], ['sentiment_report'])
            ],
            'translator': [
                AgentCapability('translator', 'text_translation', 'Translate text between languages', ['text', 'source_lang', 'target_lang'], ['translated_text']),
                AgentCapability('translator', 'language_detection', 'Detect language of text', ['text'], ['detected_language']),
                AgentCapability('translator', 'cultural_localization', 'Localize content for specific cultures', ['content', 'target_culture'], ['localized_content']),
                AgentCapability('translator', 'multilingual_seo', 'Optimize content for multiple languages', ['content'], ['seo_optimized_content'])
            ],
            'architect': [
                AgentCapability('architect', 'system_design', 'Design system architecture', ['requirements'], ['architecture_design']),
                AgentCapability('architect', 'technology_selection', 'Select appropriate technologies', ['project_requirements'], ['tech_stack']),
                AgentCapability('architect', 'scalability_planning', 'Plan system scalability', ['current_architecture'], ['scalability_plan']),
                AgentCapability('architect', 'integration_design', 'Design system integrations', ['systems'], ['integration_plan'])
            ],
            'tester': [
                AgentCapability('tester', 'test_automation', 'Automate test execution', ['test_config'], ['test_results']),
                AgentCapability('tester', 'test_strategy', 'Develop testing strategies', ['project_requirements'], ['test_strategy']),
                AgentCapability('tester', 'performance_testing', 'Execute performance tests', ['performance_config'], ['performance_results']),
                AgentCapability('tester', 'security_testing', 'Execute security tests', ['security_config'], ['security_results'])
            ]
        }
    
    async def get_or_create_agent(self, agent_type: str) -> Optional[AgentInterface]:
        """Get existing agent or create new instance"""
        if agent_type in self.agents:
            return self.agents[agent_type]
        
        if agent_type not in self.agent_classes:
            logger.error(f"Unknown agent type: {agent_type}")
            return None
        
        try:
            agent_class = self.agent_classes[agent_type]
            agent = agent_class()
            await agent.initialize()
            
            self.agents[agent_type] = agent
            logger.info(f"Created and initialized {agent_type} agent")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create {agent_type} agent: {e}")
            return None
    
    async def find_capable_agents(self, capability: str) -> List[str]:
        """Find all agents that have a specific capability"""
        capable_agents = []
        
        for agent_type, caps in self.capabilities.items():
            for cap in caps:
                if cap.capability_name == capability:
                    capable_agents.append(agent_type)
                    break
        
        return capable_agents
    
    async def route_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to most appropriate agent"""
        try:
            # Simple routing logic based on request type
            routing_map = {
                'analyze_data': 'data_science',
                'train_model': 'data_science',
                'deploy_service': 'devops',
                'create_pipeline': 'devops',
                'optimize_portfolio': 'financial',
                'assess_risk': 'financial',
                'generate_story': 'creative',
                'design_visual': 'creative',
                'moderate_content': 'social',
                'resolve_conflict': 'social',
                'translate_text': 'translator',
                'detect_language': 'translator',
                'design_system': 'architect',
                'select_technology': 'architect',
                'run_tests': 'tester',
                'create_test_strategy': 'tester'
            }
            
            agent_type = routing_map.get(request_type)
            if not agent_type:
                return {'error': f'No agent found for request type: {request_type}'}
            
            agent = await self.get_or_create_agent(agent_type)
            if not agent:
                return {'error': f'Failed to get {agent_type} agent'}
            
            # Route to appropriate method based on request type
            if hasattr(agent, 'process_request'):
                return await agent.process_request(request_data)
            else:
                # Use generate method as fallback
                prompt = f\"Process {request_type} request: {request_data}\"\n                response = await agent.generate(prompt)\n                return {'response': response, 'agent_type': agent_type}\n                \n        except Exception as e:\n            logger.error(f\"Request routing failed: {e}\")\n            return {'error': str(e)}\n    \n    async def get_agent_status(self) -> Dict[str, Any]:\n        \"\"\"Get status of all agents in registry\"\"\"\n        status = {\n            'registry_initialized': self.initialized,\n            'total_agents': len(self.agents),\n            'available_agent_types': list(self.agent_classes.keys()),\n            'active_agents': {},\n            'total_capabilities': sum(len(caps) for caps in self.capabilities.values())\n        }\n        \n        for agent_type, agent in self.agents.items():\n            try:\n                agent_status = await agent.introspect()\n                status['active_agents'][agent_type] = agent_status\n            except Exception as e:\n                status['active_agents'][agent_type] = {'error': str(e)}\n        \n        return status\n    \n    async def coordinate_multi_agent_task(self, task_description: str, required_capabilities: List[str]) -> Dict[str, Any]:\n        \"\"\"Coordinate task requiring multiple specialized agents\"\"\"\n        try:\n            # Find agents for each required capability\n            task_agents = {}\n            for capability in required_capabilities:\n                capable_agents = await self.find_capable_agents(capability)\n                if capable_agents:\n                    agent = await self.get_or_create_agent(capable_agents[0])\n                    if agent:\n                        task_agents[capability] = agent\n            \n            if len(task_agents) != len(required_capabilities):\n                return {'error': 'Could not find agents for all required capabilities'}\n            \n            # Execute coordinated task\n            results = {}\n            for capability, agent in task_agents.items():\n                try:\n                    prompt = f\"Contribute to task: {task_description}. Your role: {capability}\"\n                    result = await agent.generate(prompt)\n                    results[capability] = result\n                except Exception as e:\n                    results[capability] = {'error': str(e)}\n            \n            return {\n                'task_description': task_description,\n                'required_capabilities': required_capabilities,\n                'participating_agents': list(task_agents.keys()),\n                'results': results,\n                'coordination_success': len([r for r in results.values() if 'error' not in str(r)]) == len(required_capabilities)\n            }\n            \n        except Exception as e:\n            logger.error(f\"Multi-agent coordination failed: {e}\")\n            return {'error': str(e)}\n    \n    async def shutdown_all_agents(self):\n        \"\"\"Shutdown all active agents\"\"\"\n        try:\n            for agent_type, agent in self.agents.items():\n                try:\n                    await agent.shutdown()\n                    logger.info(f\"Shut down {agent_type} agent\")\n                except Exception as e:\n                    logger.error(f\"Error shutting down {agent_type} agent: {e}\")\n            \n            self.agents.clear()\n            self.initialized = False\n            logger.info(\"All agents shut down successfully\")\n            \n        except Exception as e:\n            logger.error(f\"Error during shutdown: {e}\")\n    \n    def get_capability_documentation(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive documentation of all agent capabilities\"\"\"\n        documentation = {\n            'overview': 'Specialized Agent Registry provides 8 specialized agents with 32+ capabilities',\n            'agent_types': {},\n            'capabilities_by_domain': {},\n            'integration_examples': []\n        }\n        \n        # Document each agent type\n        for agent_type, capabilities in self.capabilities.items():\n            documentation['agent_types'][agent_type] = {\n                'description': f'{agent_type.title()} specialist agent',\n                'capabilities': [cap.capability_name for cap in capabilities],\n                'input_types': list(set().union(*[cap.input_types for cap in capabilities])),\n                'output_types': list(set().union(*[cap.output_types for cap in capabilities]))\n            }\n        \n        # Group capabilities by domain\n        domains = {\n            'data_analysis': ['data_science'],\n            'infrastructure': ['devops', 'architect'],\n            'quality_assurance': ['tester'],\n            'business_intelligence': ['financial'],\n            'content_creation': ['creative', 'translator'],\n            'community_management': ['social']\n        }\n        \n        for domain, agent_types in domains.items():\n            domain_capabilities = []\n            for agent_type in agent_types:\n                if agent_type in self.capabilities:\n                    domain_capabilities.extend([cap.capability_name for cap in self.capabilities[agent_type]])\n            documentation['capabilities_by_domain'][domain] = domain_capabilities\n        \n        # Add integration examples\n        documentation['integration_examples'] = [\n            {\n                'scenario': 'Full Stack Development',\n                'agents': ['architect', 'devops', 'tester'],\n                'workflow': 'Design system → Deploy infrastructure → Run tests'\n            },\n            {\n                'scenario': 'Data-Driven Business Intelligence',\n                'agents': ['data_science', 'financial', 'creative'],\n                'workflow': 'Analyze data → Generate insights → Create reports'\n            },\n            {\n                'scenario': 'Global Content Localization',\n                'agents': ['translator', 'creative', 'social'],\n                'workflow': 'Translate content → Adapt creatively → Engage communities'\n            }\n        ]\n        \n        return documentation\n\n\n# Global registry instance\n_global_registry = None\n\n\nasync def get_global_registry() -> SpecializedAgentRegistry:\n    \"\"\"Get or create global agent registry instance\"\"\"\n    global _global_registry\n    if _global_registry is None:\n        _global_registry = SpecializedAgentRegistry()\n        await _global_registry.initialize()\n    return _global_registry\n\n\nasync def shutdown_global_registry():\n    \"\"\"Shutdown global registry\"\"\"\n    global _global_registry\n    if _global_registry is not None:\n        await _global_registry.shutdown_all_agents()\n        _global_registry = None