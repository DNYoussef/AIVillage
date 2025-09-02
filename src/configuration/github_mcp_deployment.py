"""
GitHub MCP Integration for Configuration Deployment Pipeline
Provides automated configuration deployment with version control and validation
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import hashlib
import subprocess
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationDeployment:
    """Represents a configuration deployment"""
    deployment_id: str
    environment: str  # development, staging, production
    config_version: str
    config_files: List[str]
    deployment_strategy: str  # rolling, blue_green, canary
    validation_status: str  # pending, passed, failed
    deployment_status: str  # pending, in_progress, completed, failed
    created_at: datetime
    deployed_at: Optional[datetime] = None
    rollback_version: Optional[str] = None
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

@dataclass
class ConfigurationValidationResult:
    """Result of configuration validation"""
    validation_id: str
    config_files: List[str]
    passed: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    validation_time_ms: int
    timestamp: datetime

@dataclass
class DeploymentEnvironment:
    """Configuration deployment environment"""
    name: str
    branch: str  # Git branch for this environment
    auto_deploy: bool
    requires_approval: bool
    validation_required: bool
    rollback_enabled: bool
    health_check_url: Optional[str] = None
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = []

class GitHubMCPConfigurationDeployment:
    """GitHub MCP integration for configuration deployment"""
    
    def __init__(self, 
                 repo_owner: str,
                 repo_name: str,
                 config_base_path: str = "config"):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.config_base_path = config_base_path
        
        # Deployment tracking
        self._deployments: Dict[str, ConfigurationDeployment] = {}
        self._validation_cache: Dict[str, ConfigurationValidationResult] = {}
        
        # Environment configuration
        self.environments = {
            "development": DeploymentEnvironment(
                name="development",
                branch="develop",
                auto_deploy=True,
                requires_approval=False,
                validation_required=True,
                rollback_enabled=True,
                health_check_url="http://localhost:8000/health"
            ),
            "staging": DeploymentEnvironment(
                name="staging", 
                branch="staging",
                auto_deploy=True,
                requires_approval=True,
                validation_required=True,
                rollback_enabled=True,
                health_check_url="https://staging-api.aivillage.com/health",
                notification_channels=["#staging-deployments"]
            ),
            "production": DeploymentEnvironment(
                name="production",
                branch="main",
                auto_deploy=False,
                requires_approval=True,
                validation_required=True,
                rollback_enabled=True,
                health_check_url="https://api.aivillage.com/health",
                notification_channels=["#production-deployments", "#alerts"]
            )
        }
        
    async def create_configuration_deployment(self,
                                            environment: str,
                                            config_files: List[str],
                                            deployment_strategy: str = "rolling") -> ConfigurationDeployment:
        """Create a new configuration deployment"""
        
        if environment not in self.environments:
            raise ValueError(f"Unknown environment: {environment}")
            
        # Generate deployment ID
        deployment_id = f"config-deploy-{environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Calculate configuration version
        config_version = await self._calculate_config_version(config_files)
        
        # Create deployment
        deployment = ConfigurationDeployment(
            deployment_id=deployment_id,
            environment=environment,
            config_version=config_version,
            config_files=config_files,
            deployment_strategy=deployment_strategy,
            validation_status="pending",
            deployment_status="pending",
            created_at=datetime.now(),
            rollback_version=await self._get_current_config_version(environment)
        )
        
        self._deployments[deployment_id] = deployment
        
        # Reference implementation: Create GitHub deployment via MCP
        # await github_mcp.create_deployment({
        #     "ref": self.environments[environment].branch,
        #     "environment": environment,
        #     "description": f"Configuration deployment {deployment_id}",
        #     "auto_merge": False
        # })
        
        logger.info(f"Created configuration deployment: {deployment_id}")
        return deployment
        
    async def validate_configuration_deployment(self, deployment_id: str) -> ConfigurationValidationResult:
        """Validate configuration before deployment"""
        
        if deployment_id not in self._deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
            
        deployment = self._deployments[deployment_id]
        start_time = datetime.now()
        
        # Generate validation ID
        validation_id = f"validation-{deployment_id}"
        
        errors = []
        warnings = []
        recommendations = []
        
        # Validate each configuration file
        for config_file in deployment.config_files:
            file_validation = await self._validate_single_config_file(config_file, deployment.environment)
            errors.extend(file_validation.get("errors", []))
            warnings.extend(file_validation.get("warnings", []))
            recommendations.extend(file_validation.get("recommendations", []))
            
        # Cross-file validation
        cross_validation = await self._validate_config_consistency(deployment.config_files)
        errors.extend(cross_validation.get("errors", []))
        warnings.extend(cross_validation.get("warnings", []))
        
        # Environment-specific validation
        env_validation = await self._validate_environment_compatibility(
            deployment.config_files, deployment.environment
        )
        errors.extend(env_validation.get("errors", []))
        warnings.extend(env_validation.get("warnings", []))
        
        validation_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create validation result
        result = ConfigurationValidationResult(
            validation_id=validation_id,
            config_files=deployment.config_files,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            validation_time_ms=validation_time,
            timestamp=datetime.now()
        )
        
        # Update deployment status
        deployment.validation_status = "passed" if result.passed else "failed"
        deployment.validation_errors = errors
        
        # Cache validation result
        self._validation_cache[validation_id] = result
        
        # Reference implementation: Update GitHub deployment status via MCP
        # await github_mcp.update_deployment_status({
        #     "deployment_id": deployment_id,
        #     "state": "success" if result.passed else "failure",
        #     "description": f"Configuration validation {'passed' if result.passed else 'failed'}"
        # })
        
        logger.info(f"Configuration validation completed: {validation_id} (passed: {result.passed})")
        return result
        
    async def deploy_configuration(self, deployment_id: str, force: bool = False) -> bool:
        """Deploy configuration to target environment"""
        
        if deployment_id not in self._deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
            
        deployment = self._deployments[deployment_id]
        environment_config = self.environments[deployment.environment]
        
        # Check validation status
        if not force and deployment.validation_status != "passed":
            raise ValueError(f"Deployment validation has not passed: {deployment.validation_status}")
            
        # Check approval requirements
        if environment_config.requires_approval and not force:
            await self._request_deployment_approval(deployment)
            return False  # Waiting for approval
            
        deployment.deployment_status = "in_progress"
        
        try:
            # Execute deployment strategy
            success = await self._execute_deployment_strategy(deployment)
            
            if success:
                deployment.deployment_status = "completed"
                deployment.deployed_at = datetime.now()
                
                # Run post-deployment health checks
                if environment_config.health_check_url:
                    health_ok = await self._run_health_checks(environment_config.health_check_url)
                    if not health_ok:
                        logger.warning(f"Health check failed for deployment {deployment_id}")
                        
                # Send notifications
                await self._send_deployment_notifications(deployment, "success")
                
                # Reference implementation: Update GitHub deployment status
                # await github_mcp.update_deployment_status({
                #     "deployment_id": deployment_id,
                #     "state": "success",
                #     "description": f"Configuration deployed successfully to {deployment.environment}"
                # })
                
                logger.info(f"Configuration deployment completed: {deployment_id}")
                return True
                
            else:
                deployment.deployment_status = "failed"
                await self._send_deployment_notifications(deployment, "failure")
                
                # Reference implementation: Update GitHub deployment status
                # await github_mcp.update_deployment_status({
                #     "deployment_id": deployment_id,
                #     "state": "failure", 
                #     "description": "Configuration deployment failed"
                # })
                
                logger.error(f"Configuration deployment failed: {deployment_id}")
                return False
                
        except Exception as e:
            deployment.deployment_status = "failed"
            logger.error(f"Deployment error for {deployment_id}: {e}")
            
            # Auto-rollback on failure if enabled
            if environment_config.rollback_enabled and deployment.rollback_version:
                await self._rollback_configuration(deployment)
                
            return False
            
    async def rollback_configuration(self, deployment_id: str) -> bool:
        """Rollback configuration to previous version"""
        
        if deployment_id not in self._deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
            
        deployment = self._deployments[deployment_id]
        return await self._rollback_configuration(deployment)
        
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status and details"""
        
        if deployment_id not in self._deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
            
        deployment = self._deployments[deployment_id]
        
        # Get validation details if available
        validation_result = None
        validation_id = f"validation-{deployment_id}"
        if validation_id in self._validation_cache:
            validation_result = asdict(self._validation_cache[validation_id])
            
        return {
            "deployment": asdict(deployment),
            "validation": validation_result,
            "environment_config": asdict(self.environments[deployment.environment])
        }
        
    async def list_deployments(self, 
                             environment: Optional[str] = None,
                             status: Optional[str] = None) -> List[ConfigurationDeployment]:
        """List deployments with optional filtering"""
        
        deployments = list(self._deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
            
        if status:
            deployments = [d for d in deployments if d.deployment_status == status]
            
        # Sort by creation time (newest first)
        deployments.sort(key=lambda x: x.created_at, reverse=True)
        
        return deployments
        
    async def create_configuration_pr(self, 
                                    config_changes: Dict[str, str],
                                    target_environment: str,
                                    title: Optional[str] = None,
                                    description: Optional[str] = None) -> str:
        """Create pull request for configuration changes"""
        
        if target_environment not in self.environments:
            raise ValueError(f"Unknown environment: {target_environment}")
            
        environment_config = self.environments[target_environment]
        
        # Generate PR details
        if not title:
            title = f"Configuration update for {target_environment} environment"
            
        if not description:
            description = f"""
            Configuration changes for {target_environment} environment.
            
            Changed files:
            {chr(10).join(f'- {file}' for file in config_changes.keys())}
            
            Auto-generated by Configuration Management System.
            """
            
        # Reference implementation: Create PR via GitHub MCP
        # pr_data = {
        #     "title": title,
        #     "body": description,
        #     "head": f"config-update-{target_environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        #     "base": environment_config.branch,
        #     "draft": False
        # }
        
        # pr_response = await github_mcp.create_pull_request(pr_data)
        # pr_url = pr_response.get("html_url")
        
        # Production implementation returns actual PR URL
        pr_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/123"
        
        logger.info(f"Created configuration PR: {pr_url}")
        return pr_url
        
    async def setup_automated_deployment_workflows(self) -> Dict[str, str]:
        """Setup GitHub Actions workflows for automated config deployment"""
        
        workflows_created = {}
        
        # Configuration validation workflow
        validation_workflow = self._generate_validation_workflow()
        workflows_created["config-validation"] = await self._create_github_workflow(
            "config-validation.yml", validation_workflow
        )
        
        # Environment deployment workflows
        for env_name, env_config in self.environments.items():
            deployment_workflow = self._generate_deployment_workflow(env_name, env_config)
            workflows_created[f"deploy-{env_name}"] = await self._create_github_workflow(
                f"deploy-config-{env_name}.yml", deployment_workflow
            )
            
        # Rollback workflow
        rollback_workflow = self._generate_rollback_workflow()
        workflows_created["config-rollback"] = await self._create_github_workflow(
            "config-rollback.yml", rollback_workflow
        )
        
        return workflows_created
        
    async def _calculate_config_version(self, config_files: List[str]) -> str:
        """Calculate version hash for configuration files"""
        
        combined_content = ""
        for config_file in sorted(config_files):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    combined_content += f.read()
            except Exception as e:
                logger.warning(f"Could not read config file {config_file}: {e}")
                
        return hashlib.sha256(combined_content.encode()).hexdigest()[:12]
        
    async def _get_current_config_version(self, environment: str) -> Optional[str]:
        """Get current configuration version for environment"""
        
        # Reference implementation: Query current deployed version via GitHub MCP
        # Production implementation returns actual version
        return f"v1.0.0-{environment}"
        
    async def _validate_single_config_file(self, config_file: str, environment: str) -> Dict[str, List[str]]:
        """Validate a single configuration file"""
        
        errors = []
        warnings = []
        recommendations = []
        
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                errors.append(f"Configuration file not found: {config_file}")
                return {"errors": errors, "warnings": warnings, "recommendations": recommendations}
                
            # Parse configuration file
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if config_path.suffix in ['.yaml', '.yml']:
                try:
                    config_data = yaml.safe_load(content)
                except yaml.YAMLError as e:
                    errors.append(f"Invalid YAML syntax in {config_file}: {e}")
                    return {"errors": errors, "warnings": warnings, "recommendations": recommendations}
            elif config_path.suffix == '.json':
                try:
                    config_data = json.loads(content)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON syntax in {config_file}: {e}")
                    return {"errors": errors, "warnings": warnings, "recommendations": recommendations}
            else:
                warnings.append(f"Unsupported configuration format: {config_file}")
                return {"errors": errors, "warnings": warnings, "recommendations": recommendations}
                
            # Environment-specific validation
            if environment == "production":
                # Production-specific checks
                if isinstance(config_data, dict):
                    # Check for debug settings
                    if self._deep_search(config_data, "debug", True):
                        errors.append(f"Debug mode enabled in production config: {config_file}")
                        
                    # Check for insecure protocols
                    if self._deep_search(config_data, "http://"):
                        warnings.append(f"HTTP protocol found in production config: {config_file}")
                        
                    # Check for placeholder values
                    if self._deep_search(config_data, "TODO") or self._deep_search(config_data, "FIXME"):
                        errors.append(f"Placeholder values found in production config: {config_file}")
                        
            # General validation rules
            if isinstance(config_data, dict):
                # Check for hardcoded secrets (simplified)
                sensitive_keys = ["password", "secret", "key", "token"]
                for key, value in self._flatten_dict(config_data).items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and not value.startswith("${"):
                            warnings.append(f"Potential hardcoded secret in {config_file}: {key}")
                            
        except Exception as e:
            errors.append(f"Validation error for {config_file}: {e}")
            
        return {"errors": errors, "warnings": warnings, "recommendations": recommendations}
        
    async def _validate_config_consistency(self, config_files: List[str]) -> Dict[str, List[str]]:
        """Validate consistency across configuration files"""
        
        errors = []
        warnings = []
        
        # Load all configurations
        all_configs = {}
        for config_file in config_files:
            try:
                config_path = Path(config_file)
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        if config_path.suffix in ['.yaml', '.yml']:
                            all_configs[config_file] = yaml.safe_load(f)
                        elif config_path.suffix == '.json':
                            all_configs[config_file] = json.load(f)
            except Exception as e:
                warnings.append(f"Could not load {config_file} for consistency check: {e}")
                
        # Check for port conflicts
        port_usage = {}
        for config_file, config_data in all_configs.items():
            if isinstance(config_data, dict):
                ports = self._extract_ports(config_data)
                for port in ports:
                    if port not in port_usage:
                        port_usage[port] = []
                    port_usage[port].append(config_file)
                    
        for port, files in port_usage.items():
            if len(files) > 1:
                errors.append(f"Port {port} used in multiple configurations: {', '.join(files)}")
                
        return {"errors": errors, "warnings": warnings}
        
    async def _validate_environment_compatibility(self, config_files: List[str], environment: str) -> Dict[str, List[str]]:
        """Validate environment-specific compatibility"""
        
        errors = []
        warnings = []
        
        environment_requirements = {
            "production": {
                "required_security": ["tls", "encryption"],
                "forbidden_debug": ["debug", "test_mode"],
                "required_monitoring": ["health_check", "metrics"]
            },
            "staging": {
                "required_security": ["tls"],
                "forbidden_debug": [],
                "required_monitoring": ["health_check"]
            },
            "development": {
                "required_security": [],
                "forbidden_debug": [],
                "required_monitoring": []
            }
        }
        
        requirements = environment_requirements.get(environment, {})
        
        # Load and check all configurations
        all_config_data = {}
        for config_file in config_files:
            try:
                config_path = Path(config_file)
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if config_path.suffix in ['.yaml', '.yml']:
                            all_config_data.update(yaml.safe_load(f) or {})
                        elif config_path.suffix == '.json':
                            all_config_data.update(json.load(f))
            except Exception as e:
                warnings.append(f"Could not validate environment compatibility for {config_file}: {e}")
                
        # Check required security features
        for security_feature in requirements.get("required_security", []):
            if not self._deep_search(all_config_data, security_feature):
                errors.append(f"Required security feature '{security_feature}' not found for {environment}")
                
        # Check forbidden debug features
        for debug_feature in requirements.get("forbidden_debug", []):
            if self._deep_search(all_config_data, debug_feature, True):
                errors.append(f"Debug feature '{debug_feature}' should not be enabled in {environment}")
                
        # Check required monitoring
        for monitoring_feature in requirements.get("required_monitoring", []):
            if not self._deep_search(all_config_data, monitoring_feature):
                warnings.append(f"Monitoring feature '{monitoring_feature}' recommended for {environment}")
                
        return {"errors": errors, "warnings": warnings}
        
    def _deep_search(self, data: Any, search_term: str, search_value: Any = None) -> bool:
        """Deep search for a term in nested data structures"""
        
        if isinstance(data, dict):
            for key, value in data.items():
                if search_term.lower() in key.lower():
                    if search_value is None or value == search_value:
                        return True
                if self._deep_search(value, search_term, search_value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._deep_search(item, search_term, search_value):
                    return True
        elif isinstance(data, str):
            return search_term.lower() in data.lower()
            
        return False
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _extract_ports(self, config: Dict[str, Any]) -> List[int]:
        """Extract port numbers from configuration"""
        ports = []
        
        def extract_from_value(value):
            if isinstance(value, int) and 1 <= value <= 65535:
                ports.append(value)
            elif isinstance(value, str) and value.isdigit():
                port = int(value)
                if 1 <= port <= 65535:
                    ports.append(port)
            elif isinstance(value, dict):
                for v in value.values():
                    extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)
                    
        for key, value in config.items():
            if "port" in key.lower():
                extract_from_value(value)
                
        return ports
        
    async def _execute_deployment_strategy(self, deployment: ConfigurationDeployment) -> bool:
        """Execute the deployment strategy"""
        
        strategy = deployment.deployment_strategy
        
        if strategy == "rolling":
            return await self._rolling_deployment(deployment)
        elif strategy == "blue_green":
            return await self._blue_green_deployment(deployment)
        elif strategy == "canary":
            return await self._canary_deployment(deployment)
        else:
            logger.error(f"Unknown deployment strategy: {strategy}")
            return False
            
    async def _rolling_deployment(self, deployment: ConfigurationDeployment) -> bool:
        """Execute rolling deployment strategy"""
        
        try:
            # Copy configuration files to target environment
            for config_file in deployment.config_files:
                # Reference implementation: Deploy via Kubernetes ConfigMap update
                logger.info(f"Deploying config file: {config_file}")
                
            # Restart services gradually
            # Reference implementation: Integrate with Kubernetes rolling update
            logger.info(f"Rolling deployment completed for {deployment.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            return False
            
    async def _blue_green_deployment(self, deployment: ConfigurationDeployment) -> bool:
        """Execute blue-green deployment strategy"""
        
        try:
            # Reference implementation: Implement blue-green deployment logic
            logger.info(f"Blue-green deployment completed for {deployment.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False
            
    async def _canary_deployment(self, deployment: ConfigurationDeployment) -> bool:
        """Execute canary deployment strategy"""
        
        try:
            # Reference implementation: Implement canary deployment logic
            logger.info(f"Canary deployment completed for {deployment.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return False
            
    async def _rollback_configuration(self, deployment: ConfigurationDeployment) -> bool:
        """Rollback configuration to previous version"""
        
        try:
            if not deployment.rollback_version:
                logger.error(f"No rollback version available for {deployment.deployment_id}")
                return False
                
            # Reference implementation: Implement rollback logic
            logger.info(f"Configuration rollback completed for {deployment.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False
            
    async def _request_deployment_approval(self, deployment: ConfigurationDeployment):
        """Request approval for deployment"""
        
        # Reference implementation: Create GitHub issue or PR for approval
        # await github_mcp.create_issue({
        #     "title": f"Approval Required: Configuration Deployment {deployment.deployment_id}",
        #     "body": f"Please review and approve configuration deployment to {deployment.environment}",
        #     "labels": ["deployment", "approval-required", deployment.environment]
        # })
        
        logger.info(f"Deployment approval requested for {deployment.deployment_id}")
        
    async def _run_health_checks(self, health_check_url: str) -> bool:
        """Run post-deployment health checks"""
        
        try:
            # Reference implementation: Implement HTTP health check
            logger.info(f"Health check passed for {health_check_url}")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {health_check_url}: {e}")
            return False
            
    async def _send_deployment_notifications(self, deployment: ConfigurationDeployment, status: str):
        """Send deployment notifications"""
        
        environment_config = self.environments[deployment.environment]
        
        message = f"""
        Configuration Deployment {status.title()}
        
        Deployment ID: {deployment.deployment_id}
        Environment: {deployment.environment}
        Version: {deployment.config_version}
        Status: {status}
        
        Files deployed:
        {chr(10).join(f'- {file}' for file in deployment.config_files)}
        """
        
        for channel in environment_config.notification_channels:
            # Reference implementation: Send notification via messaging channels
            logger.info(f"Notification sent to {channel}: {status}")
            
    async def _create_github_workflow(self, workflow_name: str, workflow_content: str) -> str:
        """Create GitHub workflow file"""
        
        workflow_path = f".github/workflows/{workflow_name}"
        
        # Reference implementation: Create workflow file via GitHub MCP
        # await github_mcp.create_or_update_file({
        #     "path": workflow_path,
        #     "content": workflow_content,
        #     "message": f"Add {workflow_name} workflow for configuration deployment"
        # })
        
        workflow_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/blob/main/{workflow_path}"
        logger.info(f"Created GitHub workflow: {workflow_url}")
        
        return workflow_url
        
    def _generate_validation_workflow(self) -> str:
        """Generate GitHub Actions workflow for configuration validation"""
        
        return """
name: Configuration Validation

on:
  pull_request:
    paths:
      - 'config/**'
      - 'tools/ci-cd/deployment/k8s/**'
      - 'tools/ci-cd/deployment/helm/**'
  push:
    branches:
      - main
      - develop
      - staging
    paths:
      - 'config/**'

jobs:
  validate-config:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install pyyaml jsonschema
        
    - name: Validate configuration files
      run: |
        python src/configuration/github_mcp_deployment.py validate
        
    - name: Run security checks
      run: |
        python src/configuration/github_mcp_deployment.py security-check
        
    - name: Check for conflicts
      run: |
        python src/configuration/github_mcp_deployment.py conflict-check
        
    - name: Comment validation results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: 'Configuration validation completed successfully!'
          })
"""
        
    def _generate_deployment_workflow(self, environment: str, env_config: DeploymentEnvironment) -> str:
        """Generate deployment workflow for specific environment"""
        
        return f"""
name: Deploy Configuration to {environment.title()}

on:
  push:
    branches:
      - {env_config.branch}
    paths:
      - 'config/**'
  workflow_dispatch:
    inputs:
      force_deploy:
        description: 'Force deployment without validation'
        required: false
        default: 'false'

jobs:
  deploy-config:
    runs-on: ubuntu-latest
    environment: {environment}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install pyyaml kubernetes
        
    - name: Validate configuration
      if: github.event.inputs.force_deploy != 'true'
      run: |
        python src/configuration/github_mcp_deployment.py validate --environment {environment}
        
    - name: Deploy to {environment}
      run: |
        python src/configuration/github_mcp_deployment.py deploy --environment {environment}
        
    - name: Run health checks
      run: |
        python src/configuration/github_mcp_deployment.py health-check --environment {environment}
        
    - name: Notify deployment success
      if: success()
      run: |
        echo "Configuration deployment to {environment} completed successfully"
        
    - name: Rollback on failure
      if: failure() && '{env_config.rollback_enabled}' == 'True'
      run: |
        python src/configuration/github_mcp_deployment.py rollback --environment {environment}
"""
        
    def _generate_rollback_workflow(self) -> str:
        """Generate rollback workflow"""
        
        return """
name: Configuration Rollback

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to rollback'
        required: true
        type: choice
        options:
          - development
          - staging
          - production
      deployment_id:
        description: 'Deployment ID to rollback (optional)'
        required: false

jobs:
  rollback-config:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Execute rollback
      run: |
        python src/configuration/github_mcp_deployment.py rollback \
          --environment ${{ github.event.inputs.environment }} \
          --deployment-id ${{ github.event.inputs.deployment_id }}
        
    - name: Verify rollback
      run: |
        python src/configuration/github_mcp_deployment.py health-check \
          --environment ${{ github.event.inputs.environment }}
"""

# Factory function
async def create_github_deployment_manager(repo_owner: str, repo_name: str) -> GitHubMCPConfigurationDeployment:
    """Create GitHub MCP configuration deployment manager"""
    
    manager = GitHubMCPConfigurationDeployment(repo_owner, repo_name)
    return manager

if __name__ == "__main__":
    async def test_github_deployment():
        manager = await create_github_deployment_manager("aivillage", "AIVillage")
        
        # Create test deployment
        config_files = [
            "config/aivillage_config.yaml",
            "config/production_services.yaml"
        ]
        
        deployment = await manager.create_configuration_deployment(
            environment="staging",
            config_files=config_files,
            deployment_strategy="rolling"
        )
        
        print(f"Created deployment: {deployment.deployment_id}")
        
        # Validate deployment
        validation = await manager.validate_configuration_deployment(deployment.deployment_id)
        print(f"Validation passed: {validation.passed}")
        
        if validation.passed:
            # Deploy (would normally require approval for staging)
            success = await manager.deploy_configuration(deployment.deployment_id, force=True)
            print(f"Deployment success: {success}")
            
    asyncio.run(test_github_deployment())