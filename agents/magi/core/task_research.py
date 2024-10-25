"""Task research functionality for MagiAgent."""

import logging
from typing import Dict, Any, List, Optional
from .research_integration import ResearchIntegration

logger = logging.getLogger(__name__)

class TaskResearch:
    """
    Handles research-based task planning for MagiAgent.
    """
    
    def __init__(
        self,
        github_token: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        llm: Any = None
    ):
        self.research_integration = ResearchIntegration(
            github_token=github_token,
            huggingface_token=huggingface_token,
            llm=llm
        )

    async def plan_task_with_research(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan task implementation using research.

        :param task: Task description and requirements
        :return: Implementation plan with resources
        """
        # Perform research
        research_results = await self.research_integration.research_for_task(task)
        
        # Analyze implementation approaches
        implementation_analysis = await self.research_integration.analyze_implementation_approach(
            task, research_results['research_results'])
        
        # Download required resources
        resources = await self.research_integration.download_resources(
            research_results['research_results'][0],  # Use first result set
            research_results['recommendations']
        )
        
        # Generate implementation plan
        plan = await self._generate_implementation_plan(
            task,
            research_results,
            implementation_analysis,
            resources
        )
        
        return {
            'task': task,
            'research': research_results,
            'analysis': implementation_analysis,
            'resources': resources,
            'plan': plan
        }

    async def _generate_implementation_plan(
        self,
        task: Dict[str, Any],
        research_results: Dict[str, Any],
        implementation_analysis: Dict[str, Any],
        resources: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate detailed implementation plan."""
        plan_prompt = f"""
        Generate detailed implementation plan:
        
        Task: {task}
        Research Analysis: {research_results['analysis']}
        Implementation Analysis: {implementation_analysis['analysis']}
        Available Resources: {list(resources.keys())}
        
        Provide:
        1. Step-by-step implementation steps
        2. Required tools and resources for each step
        3. Integration points with downloaded resources
        4. Testing requirements
        5. Success criteria
        """
        
        plan_text = await self.research_integration.research_manager.llm.generate(
            plan_prompt)
            
        # Parse plan into structured format
        steps = []
        tools_needed = set()
        current_section = None
        current_step = None
        
        for line in plan_text.split('\n'):
            if 'Step ' in line:
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'description': line.split('Step ')[1],
                    'tools': [],
                    'resources': [],
                    'tests': []
                }
            elif 'Tools:' in line:
                current_section = 'tools'
            elif 'Resources:' in line:
                current_section = 'resources'
            elif 'Tests:' in line:
                current_section = 'tests'
            elif line.strip() and current_step and current_section:
                if current_section == 'tools':
                    tool = line.strip()
                    current_step['tools'].append(tool)
                    tools_needed.add(tool)
                elif current_section == 'resources':
                    current_step['resources'].append(line.strip())
                elif current_section == 'tests':
                    current_step['tests'].append(line.strip())
                    
        if current_step:
            steps.append(current_step)
            
        return {
            'steps': steps,
            'tools_needed': list(tools_needed),
            'success_criteria': research_results['recommendations']['testing_steps'],
            'integration_steps': research_results['recommendations']['integration_steps']
        }

    async def validate_resources(
        self,
        resources: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Validate downloaded resources.

        :param resources: Dictionary of resource paths
        :return: Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        for resource_type, path in resources.items():
            try:
                if resource_type.startswith('github/'):
                    # Validate GitHub repository
                    validation_results.update(
                        await self._validate_github_repo(path))
                elif resource_type.startswith('huggingface/'):
                    # Validate Hugging Face model
                    validation_results.update(
                        await self._validate_huggingface_model(path))
            except Exception as e:
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Failed to validate {resource_type}: {str(e)}")
                
        return validation_results

    async def _validate_github_repo(self, path: str) -> Dict[str, Any]:
        """Validate GitHub repository structure and contents."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check basic repository structure
        required_files = ['README.md', 'requirements.txt', 'setup.py']
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                validation['warnings'].append(f"Missing {file}")
                
        # Check for security issues
        security_issues = await self._check_security_issues(path)
        if security_issues:
            validation['issues'].extend(security_issues)
            validation['is_valid'] = False
            
        return validation

    async def _validate_huggingface_model(self, path: str) -> Dict[str, Any]:
        """Validate Hugging Face model files and structure."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check model files
        required_files = ['config.json', 'pytorch_model.bin']
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                validation['issues'].append(f"Missing required file: {file}")
                validation['is_valid'] = False
                
        return validation

    async def _check_security_issues(self, path: str) -> List[str]:
        """Check for security issues in code."""
        issues = []
        
        # Patterns to check for
        dangerous_patterns = [
            'os.system(',
            'subprocess.call(',
            'eval(',
            'exec('
        ]
        
        # Check all Python files
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in dangerous_patterns:
                            if pattern in content:
                                issues.append(
                                    f"Dangerous pattern {pattern} found in {file}")
                                
        return issues
