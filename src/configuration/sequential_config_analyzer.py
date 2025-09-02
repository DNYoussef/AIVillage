"""
Sequential Configuration Analysis with MCP Integration
Uses Sequential Thinking MCP for systematic configuration hierarchy analysis
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationAnalysisStep:
    """Represents a step in sequential configuration analysis"""
    step_number: int
    step_name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class ConfigurationHierarchyAnalysis:
    """Complete analysis of configuration hierarchy"""
    analysis_id: str
    timestamp: datetime
    total_configs: int
    hierarchy_levels: Dict[str, int]
    conflicts: List[Dict[str, Any]]
    redundancies: List[Dict[str, Any]]
    missing_configs: List[str]
    security_issues: List[Dict[str, Any]]
    performance_recommendations: List[str]
    consolidation_plan: Dict[str, Any]
    steps: List[ConfigurationAnalysisStep]

class SequentialConfigurationAnalyzer:
    """Sequential configuration analyzer with MCP integration"""
    
    def __init__(self):
        self._analysis_cache: Dict[str, ConfigurationHierarchyAnalysis] = {}
        self._sequential_thinking_enabled = True
        
    async def analyze_configuration_hierarchy(self, 
                                            config_paths: List[str],
                                            analysis_name: str = "config_hierarchy") -> ConfigurationHierarchyAnalysis:
        """Perform sequential analysis of configuration hierarchy"""
        
        analysis_id = f"{analysis_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting sequential configuration analysis: {analysis_id}")
        
        analysis_steps = []
        
        # Step 1: Discovery and Inventory
        step1_result = await self._step_1_discovery(config_paths)
        analysis_steps.append(step1_result)
        
        if not step1_result.success:
            return self._create_failed_analysis(analysis_id, start_time, analysis_steps)
            
        discovered_configs = step1_result.outputs["discovered_configs"]
        
        # Step 2: Hierarchy Mapping
        step2_result = await self._step_2_hierarchy_mapping(discovered_configs)
        analysis_steps.append(step2_result)
        
        # Step 3: Conflict Detection
        step3_result = await self._step_3_conflict_detection(discovered_configs)
        analysis_steps.append(step3_result)
        
        # Step 4: Redundancy Analysis
        step4_result = await self._step_4_redundancy_analysis(discovered_configs)
        analysis_steps.append(step4_result)
        
        # Step 5: Security Assessment
        step5_result = await self._step_5_security_assessment(discovered_configs)
        analysis_steps.append(step5_result)
        
        # Step 6: Performance Analysis
        step6_result = await self._step_6_performance_analysis(discovered_configs)
        analysis_steps.append(step6_result)
        
        # Step 7: Consolidation Planning
        step7_result = await self._step_7_consolidation_planning(
            discovered_configs,
            step3_result.outputs.get("conflicts", []),
            step4_result.outputs.get("redundancies", [])
        )
        analysis_steps.append(step7_result)
        
        # Create comprehensive analysis
        analysis = ConfigurationHierarchyAnalysis(
            analysis_id=analysis_id,
            timestamp=start_time,
            total_configs=len(discovered_configs),
            hierarchy_levels=step2_result.outputs.get("hierarchy_levels", {}),
            conflicts=step3_result.outputs.get("conflicts", []),
            redundancies=step4_result.outputs.get("redundancies", []),
            missing_configs=step2_result.outputs.get("missing_configs", []),
            security_issues=step5_result.outputs.get("security_issues", []),
            performance_recommendations=step6_result.outputs.get("recommendations", []),
            consolidation_plan=step7_result.outputs.get("consolidation_plan", {}),
            steps=analysis_steps
        )
        
        # Cache analysis for future reference
        self._analysis_cache[analysis_id] = analysis
        
        # Reference implementation: Store in Memory MCP for pattern learning
        # await memory_mcp.store(f"config-analysis/{analysis_id}", asdict(analysis))
        
        logger.info(f"Configuration analysis completed: {analysis_id}")
        return analysis
        
    async def _step_1_discovery(self, config_paths: List[str]) -> ConfigurationAnalysisStep:
        """Step 1: Discover and inventory all configuration files"""
        start_time = datetime.now()
        step_name = "Configuration Discovery"
        
        try:
            discovered_configs = []
            
            for config_path in config_paths:
                path_obj = Path(config_path)
                
                if path_obj.is_file():
                    config_info = await self._analyze_single_config_file(path_obj)
                    if config_info:
                        discovered_configs.append(config_info)
                        
                elif path_obj.is_dir():
                    # Recursively discover config files
                    for pattern in ["*.yaml", "*.yml", "*.json", "*.env*"]:
                        for file_path in path_obj.rglob(pattern):
                            if file_path.is_file():
                                config_info = await self._analyze_single_config_file(file_path)
                                if config_info:
                                    discovered_configs.append(config_info)
                                    
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=1,
                step_name=step_name,
                description="Discover and catalog all configuration files in the system",
                inputs={"config_paths": config_paths},
                outputs={
                    "discovered_configs": discovered_configs,
                    "total_files": len(discovered_configs)
                },
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 1 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=1,
                step_name=step_name,
                description="Discover and catalog all configuration files in the system",
                inputs={"config_paths": config_paths},
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _step_2_hierarchy_mapping(self, discovered_configs: List[Dict]) -> ConfigurationAnalysisStep:
        """Step 2: Map configuration hierarchy and determine priority levels"""
        start_time = datetime.now()
        step_name = "Hierarchy Mapping"
        
        try:
            hierarchy_levels = {
                "base": 0,
                "environment": 0,
                "service": 0,
                "runtime": 0,
                "override": 0
            }
            
            missing_configs = []
            config_priorities = {}
            
            for config in discovered_configs:
                file_path = config["path"]
                
                # Determine hierarchy level based on path and content
                if "base" in file_path or "default" in file_path:
                    level = "base"
                    priority = 10
                elif any(env in file_path for env in ["development", "staging", "production"]):
                    level = "environment" 
                    priority = 30
                elif any(svc in file_path for svc in ["service", "orchestration", "rag", "hyperag"]):
                    level = "service"
                    priority = 40
                elif "k8s" in file_path or "deployment" in file_path:
                    level = "runtime"
                    priority = 50
                else:
                    level = "override"
                    priority = 60
                    
                hierarchy_levels[level] += 1
                config_priorities[file_path] = {"level": level, "priority": priority}
                
            # Check for missing essential configurations
            required_configs = [
                "database configuration",
                "api gateway configuration", 
                "security configuration",
                "logging configuration"
            ]
            
            # Simplified check - in real implementation would be more sophisticated
            for req_config in required_configs:
                found = any(req_config.split()[0] in str(config["content"]).lower() 
                          for config in discovered_configs)
                if not found:
                    missing_configs.append(req_config)
                    
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=2,
                step_name=step_name,
                description="Map configuration files to hierarchy levels and determine priorities",
                inputs={"discovered_configs": len(discovered_configs)},
                outputs={
                    "hierarchy_levels": hierarchy_levels,
                    "config_priorities": config_priorities,
                    "missing_configs": missing_configs
                },
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 2 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=2,
                step_name=step_name,
                description="Map configuration files to hierarchy levels and determine priorities",
                inputs={"discovered_configs": len(discovered_configs)},
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _step_3_conflict_detection(self, discovered_configs: List[Dict]) -> ConfigurationAnalysisStep:
        """Step 3: Detect configuration conflicts and inconsistencies"""
        start_time = datetime.now()
        step_name = "Conflict Detection"
        
        try:
            conflicts = []
            
            # Build a map of all configuration keys and their values across files
            key_value_map = {}
            
            for config in discovered_configs:
                file_path = config["path"]
                content = config.get("content", {})
                
                # Flatten configuration to detect conflicts
                flat_config = self._flatten_dict(content)
                
                for key, value in flat_config.items():
                    if key not in key_value_map:
                        key_value_map[key] = []
                    key_value_map[key].append({
                        "file": file_path,
                        "value": value
                    })
                    
            # Detect conflicts (same key, different values)
            for key, value_list in key_value_map.items():
                if len(value_list) > 1:
                    unique_values = set(str(item["value"]) for item in value_list)
                    if len(unique_values) > 1:
                        conflicts.append({
                            "key": key,
                            "conflicting_values": value_list,
                            "severity": self._assess_conflict_severity(key, value_list)
                        })
                        
            # Detect port conflicts specifically
            port_conflicts = []
            port_map = {}
            
            for config in discovered_configs:
                content = config.get("content", {})
                ports = self._extract_ports(content)
                
                for port in ports:
                    if port not in port_map:
                        port_map[port] = []
                    port_map[port].append(config["path"])
                    
            for port, files in port_map.items():
                if len(files) > 1:
                    port_conflicts.append({
                        "port": port,
                        "files": files,
                        "severity": "high"
                    })
                    
            all_conflicts = conflicts + port_conflicts
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=3,
                step_name=step_name,
                description="Detect conflicts and inconsistencies in configuration values",
                inputs={"config_files": len(discovered_configs)},
                outputs={
                    "conflicts": all_conflicts,
                    "total_conflicts": len(all_conflicts),
                    "port_conflicts": port_conflicts
                },
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 3 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=3,
                step_name=step_name,
                description="Detect conflicts and inconsistencies in configuration values",
                inputs={"config_files": len(discovered_configs)},
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _step_4_redundancy_analysis(self, discovered_configs: List[Dict]) -> ConfigurationAnalysisStep:
        """Step 4: Analyze configuration redundancies and duplication"""
        start_time = datetime.now()
        step_name = "Redundancy Analysis"
        
        try:
            redundancies = []
            
            # Group configurations by similarity
            similarity_groups = {}
            
            for i, config1 in enumerate(discovered_configs):
                for j, config2 in enumerate(discovered_configs[i+1:], i+1):
                    similarity = self._calculate_config_similarity(
                        config1.get("content", {}),
                        config2.get("content", {})
                    )
                    
                    if similarity > 0.7:  # 70% similarity threshold
                        group_key = f"group_{len(similarity_groups)}"
                        if not any(config1["path"] in group or config2["path"] in group 
                                 for group in similarity_groups.values()):
                            similarity_groups[group_key] = {
                                "files": [config1["path"], config2["path"]],
                                "similarity": similarity,
                                "redundant_keys": self._find_redundant_keys(
                                    config1.get("content", {}),
                                    config2.get("content", {})
                                )
                            }
                            
            # Convert to redundancy list
            for group_id, group_info in similarity_groups.items():
                redundancies.append({
                    "type": "duplicate_configuration",
                    "files": group_info["files"],
                    "similarity_score": group_info["similarity"],
                    "redundant_keys": group_info["redundant_keys"],
                    "recommendation": "Consider consolidating these configurations"
                })
                
            # Detect specific redundancy patterns
            database_configs = []
            api_configs = []
            
            for config in discovered_configs:
                content = str(config.get("content", {})).lower()
                if "database" in content or "db" in content:
                    database_configs.append(config["path"])
                if "api" in content or "port" in content:
                    api_configs.append(config["path"])
                    
            if len(database_configs) > 2:
                redundancies.append({
                    "type": "scattered_database_config",
                    "files": database_configs,
                    "recommendation": "Centralize database configuration"
                })
                
            if len(api_configs) > 3:
                redundancies.append({
                    "type": "scattered_api_config", 
                    "files": api_configs,
                    "recommendation": "Centralize API configuration"
                })
                
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=4,
                step_name=step_name,
                description="Analyze configuration redundancies and identify consolidation opportunities",
                inputs={"config_files": len(discovered_configs)},
                outputs={
                    "redundancies": redundancies,
                    "total_redundancies": len(redundancies),
                    "similarity_groups": len(similarity_groups)
                },
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 4 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=4,
                step_name=step_name,
                description="Analyze configuration redundancies and identify consolidation opportunities",
                inputs={"config_files": len(discovered_configs)},
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _step_5_security_assessment(self, discovered_configs: List[Dict]) -> ConfigurationAnalysisStep:
        """Step 5: Assess configuration security issues"""
        start_time = datetime.now()
        step_name = "Security Assessment"
        
        try:
            security_issues = []
            
            # Security patterns to check
            security_patterns = {
                "hardcoded_passwords": [r"password\s*=\s*['\"][^'\"]+['\"]", r"pwd\s*=\s*['\"][^'\"]+['\"]"],
                "exposed_secrets": [r"secret\s*=\s*['\"][^'\"]+['\"]", r"api_key\s*=\s*['\"][^'\"]+['\"]"],
                "insecure_protocols": ["http://", "ftp://", "telnet://"],
                "weak_encryption": ["md5", "sha1", "des"],
                "debug_enabled": [r"debug\s*=\s*true", r"DEBUG\s*=\s*['\"]?true['\"]?"]
            }
            
            for config in discovered_configs:
                file_path = config["path"]
                content_str = str(config.get("content", {}))
                
                for issue_type, patterns in security_patterns.items():
                    for pattern in patterns:
                        import re
                        if re.search(pattern, content_str, re.IGNORECASE):
                            security_issues.append({
                                "type": issue_type,
                                "file": file_path,
                                "pattern": pattern,
                                "severity": self._assess_security_severity(issue_type),
                                "recommendation": self._get_security_recommendation(issue_type)
                            })
                            
            # Check for missing security configurations
            security_checks = [
                {"key": "tls", "enabled": False, "missing_count": 0},
                {"key": "authentication", "enabled": False, "missing_count": 0},
                {"key": "encryption", "enabled": False, "missing_count": 0}
            ]
            
            for config in discovered_configs:
                content = config.get("content", {})
                content_str = str(content).lower()
                
                if "tls" in content_str or "ssl" in content_str:
                    security_checks[0]["enabled"] = True
                if "auth" in content_str or "login" in content_str:
                    security_checks[1]["enabled"] = True  
                if "encrypt" in content_str or "cipher" in content_str:
                    security_checks[2]["enabled"] = True
                    
            for check in security_checks:
                if not check["enabled"]:
                    security_issues.append({
                        "type": "missing_security_config",
                        "missing_feature": check["key"],
                        "severity": "medium",
                        "recommendation": f"Enable {check['key']} security features"
                    })
                    
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=5,
                step_name=step_name,
                description="Assess security vulnerabilities and risks in configurations",
                inputs={"config_files": len(discovered_configs)},
                outputs={
                    "security_issues": security_issues,
                    "total_issues": len(security_issues),
                    "high_severity": len([issue for issue in security_issues if issue.get("severity") == "high"])
                },
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 5 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=5,
                step_name=step_name,
                description="Assess security vulnerabilities and risks in configurations",
                inputs={"config_files": len(discovered_configs)},
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _step_6_performance_analysis(self, discovered_configs: List[Dict]) -> ConfigurationAnalysisStep:
        """Step 6: Analyze performance implications of configurations"""
        start_time = datetime.now()
        step_name = "Performance Analysis"
        
        try:
            recommendations = []
            
            # Performance analysis patterns
            performance_checks = {
                "caching": {"enabled": False, "count": 0},
                "connection_pooling": {"enabled": False, "count": 0}, 
                "resource_limits": {"enabled": False, "count": 0},
                "optimization": {"enabled": False, "count": 0}
            }
            
            for config in discovered_configs:
                content = config.get("content", {})
                content_str = str(content).lower()
                
                # Check for caching configuration
                if any(term in content_str for term in ["cache", "redis", "memcache"]):
                    performance_checks["caching"]["enabled"] = True
                    performance_checks["caching"]["count"] += 1
                    
                # Check for connection pooling
                if any(term in content_str for term in ["pool", "connection_pool", "max_connections"]):
                    performance_checks["connection_pooling"]["enabled"] = True
                    performance_checks["connection_pooling"]["count"] += 1
                    
                # Check for resource limits
                if any(term in content_str for term in ["memory", "cpu", "timeout", "limit"]):
                    performance_checks["resource_limits"]["enabled"] = True
                    performance_checks["resource_limits"]["count"] += 1
                    
                # Check for optimization settings
                if any(term in content_str for term in ["optimization", "performance", "batch_size"]):
                    performance_checks["optimization"]["enabled"] = True
                    performance_checks["optimization"]["count"] += 1
                    
            # Generate recommendations
            if not performance_checks["caching"]["enabled"]:
                recommendations.append("Enable caching to improve response times")
                
            if performance_checks["caching"]["count"] > 3:
                recommendations.append("Consider consolidating caching configuration")
                
            if not performance_checks["connection_pooling"]["enabled"]:
                recommendations.append("Implement connection pooling for database connections")
                
            if performance_checks["resource_limits"]["count"] < 3:
                recommendations.append("Add resource limits to prevent resource exhaustion")
                
            if not performance_checks["optimization"]["enabled"]:
                recommendations.append("Add performance optimization settings")
                
            # Check for configuration loading performance
            total_config_size = sum(len(str(config.get("content", {}))) for config in discovered_configs)
            if total_config_size > 1000000:  # 1MB threshold
                recommendations.append("Large configuration size detected - consider splitting or optimizing")
                
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=6,
                step_name=step_name,
                description="Analyze performance implications and generate optimization recommendations",
                inputs={"config_files": len(discovered_configs)},
                outputs={
                    "recommendations": recommendations,
                    "performance_checks": performance_checks,
                    "total_config_size_bytes": total_config_size
                },
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 6 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=6,
                step_name=step_name,
                description="Analyze performance implications and generate optimization recommendations",
                inputs={"config_files": len(discovered_configs)},
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _step_7_consolidation_planning(self, 
                                           discovered_configs: List[Dict],
                                           conflicts: List[Dict],
                                           redundancies: List[Dict]) -> ConfigurationAnalysisStep:
        """Step 7: Create configuration consolidation plan"""
        start_time = datetime.now()
        step_name = "Consolidation Planning"
        
        try:
            consolidation_plan = {
                "strategy": "hierarchical_consolidation",
                "phases": [],
                "target_structure": {},
                "migration_steps": [],
                "rollback_plan": {},
                "success_criteria": []
            }
            
            # Phase 1: Base Configuration Consolidation
            base_configs = [c for c in discovered_configs 
                          if "base" in c["path"] or "default" in c["path"]]
            
            consolidation_plan["phases"].append({
                "phase": 1,
                "name": "Base Configuration Consolidation",
                "description": "Consolidate base/default configurations into unified base layer",
                "target_files": ["config/base/unified_base_config.yaml"],
                "source_files": [c["path"] for c in base_configs],
                "priority": "high"
            })
            
            # Phase 2: Environment-Specific Consolidation  
            env_configs = [c for c in discovered_configs 
                         if any(env in c["path"] for env in ["development", "staging", "production"])]
            
            consolidation_plan["phases"].append({
                "phase": 2,
                "name": "Environment Configuration Consolidation",
                "description": "Organize environment-specific overrides",
                "target_files": [
                    "config/environments/development.yaml",
                    "config/environments/staging.yaml", 
                    "config/environments/production.yaml"
                ],
                "source_files": [c["path"] for c in env_configs],
                "priority": "high"
            })
            
            # Phase 3: Service Configuration Consolidation
            service_configs = [c for c in discovered_configs 
                             if any(svc in c["path"] for svc in ["service", "api", "rag", "hyperag"])]
            
            consolidation_plan["phases"].append({
                "phase": 3,
                "name": "Service Configuration Consolidation",
                "description": "Group service-specific configurations",
                "target_files": [
                    "config/services/gateway.yaml",
                    "config/services/agent_forge.yaml",
                    "config/services/rag.yaml",
                    "config/services/p2p.yaml"
                ],
                "source_files": [c["path"] for c in service_configs],
                "priority": "medium"
            })
            
            # Target structure
            consolidation_plan["target_structure"] = {
                "config/": {
                    "base/": {
                        "unified_base_config.yaml": "Core system defaults"
                    },
                    "environments/": {
                        "development.yaml": "Development overrides",
                        "staging.yaml": "Staging overrides",
                        "production.yaml": "Production overrides"
                    },
                    "services/": {
                        "gateway.yaml": "API Gateway configuration",
                        "agent_forge.yaml": "Agent Forge configuration",
                        "rag.yaml": "RAG pipeline configuration",
                        "p2p.yaml": "P2P networking configuration"
                    },
                    "security/": {
                        "secrets.yaml": "Encrypted secrets",
                        "policies.yaml": "Security policies"
                    }
                }
            }
            
            # Migration steps
            consolidation_plan["migration_steps"] = [
                {
                    "step": 1,
                    "action": "backup_existing_configs",
                    "description": "Create backup of all existing configuration files"
                },
                {
                    "step": 2, 
                    "action": "create_unified_manager",
                    "description": "Deploy unified configuration management system"
                },
                {
                    "step": 3,
                    "action": "migrate_base_configs",
                    "description": "Migrate base configurations to unified structure"
                },
                {
                    "step": 4,
                    "action": "migrate_environment_configs", 
                    "description": "Migrate environment-specific configurations"
                },
                {
                    "step": 5,
                    "action": "migrate_service_configs",
                    "description": "Migrate service-specific configurations"
                },
                {
                    "step": 6,
                    "action": "validate_consolidated_configs",
                    "description": "Validate all consolidated configurations"
                },
                {
                    "step": 7,
                    "action": "update_service_references",
                    "description": "Update all service references to use unified manager"
                }
            ]
            
            # Success criteria
            consolidation_plan["success_criteria"] = [
                "All services successfully load configurations via unified manager",
                "Configuration conflicts resolved",
                "Redundant configurations eliminated",
                "Security issues addressed",
                "Performance improved by at least 20%",
                "Hot-reload functionality working",
                "Distributed caching operational"
            ]
            
            # Rollback plan
            consolidation_plan["rollback_plan"] = {
                "trigger_conditions": [
                    "Service startup failures",
                    "Configuration validation errors",
                    "Performance degradation > 10%"
                ],
                "rollback_steps": [
                    "Switch services back to original config files",
                    "Restore backup configurations",
                    "Disable unified configuration manager",
                    "Verify system functionality"
                ],
                "estimated_rollback_time": "15 minutes"
            }
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConfigurationAnalysisStep(
                step_number=7,
                step_name=step_name,
                description="Create comprehensive configuration consolidation plan",
                inputs={
                    "total_configs": len(discovered_configs),
                    "conflicts": len(conflicts),
                    "redundancies": len(redundancies)
                },
                outputs={"consolidation_plan": consolidation_plan},
                duration_ms=int(duration),
                success=True
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Step 7 failed: {e}")
            
            return ConfigurationAnalysisStep(
                step_number=7,
                step_name=step_name,
                description="Create comprehensive configuration consolidation plan",
                inputs={
                    "total_configs": len(discovered_configs),
                    "conflicts": len(conflicts), 
                    "redundancies": len(redundancies)
                },
                outputs={},
                duration_ms=int(duration),
                success=False,
                error_message=str(e)
            )
            
    async def _analyze_single_config_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse based on file extension
            parsed_content = {}
            if file_path.suffix in ['.yaml', '.yml']:
                parsed_content = yaml.safe_load(content) or {}
            elif file_path.suffix == '.json':
                parsed_content = json.loads(content)
            elif '.env' in file_path.name:
                # Simple env file parsing
                parsed_content = {}
                for line in content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        parsed_content[key.strip()] = value.strip()
                        
            return {
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": len(content),
                "type": file_path.suffix,
                "content": parsed_content,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze config file {file_path}: {e}")
            return None
            
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
        
    def _assess_conflict_severity(self, key: str, value_list: List[Dict]) -> str:
        """Assess severity of configuration conflict"""
        if any(term in key.lower() for term in ["port", "url", "host", "password", "secret"]):
            return "high"
        elif any(term in key.lower() for term in ["timeout", "limit", "size"]):
            return "medium"
        else:
            return "low"
            
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
        
    def _calculate_config_similarity(self, config1: Dict, config2: Dict) -> float:
        """Calculate similarity between two configurations"""
        flat1 = self._flatten_dict(config1)
        flat2 = self._flatten_dict(config2)
        
        all_keys = set(flat1.keys()) | set(flat2.keys())
        if not all_keys:
            return 0.0
            
        common_keys = set(flat1.keys()) & set(flat2.keys())
        return len(common_keys) / len(all_keys)
        
    def _find_redundant_keys(self, config1: Dict, config2: Dict) -> List[str]:
        """Find keys that exist in both configurations with same values"""
        flat1 = self._flatten_dict(config1)
        flat2 = self._flatten_dict(config2)
        
        redundant = []
        for key in flat1.keys():
            if key in flat2 and flat1[key] == flat2[key]:
                redundant.append(key)
                
        return redundant
        
    def _assess_security_severity(self, issue_type: str) -> str:
        """Assess severity of security issue"""
        high_severity = ["hardcoded_passwords", "exposed_secrets"]
        medium_severity = ["insecure_protocols", "debug_enabled"]
        
        if issue_type in high_severity:
            return "high"
        elif issue_type in medium_severity:
            return "medium"
        else:
            return "low"
            
    def _get_security_recommendation(self, issue_type: str) -> str:
        """Get security recommendation for issue type"""
        recommendations = {
            "hardcoded_passwords": "Use environment variables or secure vaults for passwords",
            "exposed_secrets": "Move secrets to encrypted storage or environment variables",
            "insecure_protocols": "Use HTTPS/TLS instead of insecure protocols",
            "weak_encryption": "Upgrade to stronger encryption algorithms",
            "debug_enabled": "Disable debug mode in production environments"
        }
        return recommendations.get(issue_type, "Review and secure this configuration")
        
    def _create_failed_analysis(self, analysis_id: str, start_time: datetime, steps: List[ConfigurationAnalysisStep]) -> ConfigurationHierarchyAnalysis:
        """Create a failed analysis result"""
        return ConfigurationHierarchyAnalysis(
            analysis_id=analysis_id,
            timestamp=start_time,
            total_configs=0,
            hierarchy_levels={},
            conflicts=[],
            redundancies=[],
            missing_configs=[],
            security_issues=[],
            performance_recommendations=[],
            consolidation_plan={},
            steps=steps
        )

# Integration with Sequential Thinking MCP
class SequentialThinkingMCPIntegration:
    """Integration with Sequential Thinking MCP for enhanced analysis"""
    
    def __init__(self, analyzer: SequentialConfigurationAnalyzer):
        self.analyzer = analyzer
        
    async def enhanced_sequential_analysis(self, config_paths: List[str]) -> ConfigurationHierarchyAnalysis:
        """Perform enhanced analysis using Sequential Thinking MCP"""
        
        # Reference implementation: Integrate with Sequential Thinking MCP
        # reasoning_chain = await sequential_thinking_mcp.create_chain(
        #     domain="configuration-architecture",
        #     problem="Analyze and consolidate configuration hierarchy",
        #     steps=7
        # )
        
        # For now, use the standard analyzer
        analysis = await self.analyzer.analyze_configuration_hierarchy(config_paths)
        
        # Reference implementation: Enhance with MCP reasoning
        # enhanced_analysis = await sequential_thinking_mcp.enhance_analysis(
        #     analysis=analysis,
        #     reasoning_chain=reasoning_chain
        # )
        
        return analysis

if __name__ == "__main__":
    async def test_sequential_analysis():
        analyzer = SequentialConfigurationAnalyzer()
        
        config_paths = [
            "config/",
            "tools/ci-cd/deployment/k8s/",
            "tools/ci-cd/deployment/helm/"
        ]
        
        analysis = await analyzer.analyze_configuration_hierarchy(config_paths)
        
        print(f"Analysis ID: {analysis.analysis_id}")
        print(f"Total configs: {analysis.total_configs}")
        print(f"Conflicts: {len(analysis.conflicts)}")
        print(f"Redundancies: {len(analysis.redundancies)}")
        print(f"Security issues: {len(analysis.security_issues)}")
        
        print("\nConsolidation Plan:")
        for phase in analysis.consolidation_plan.get("phases", []):
            print(f"  Phase {phase['phase']}: {phase['name']}")
            
    asyncio.run(test_sequential_analysis())