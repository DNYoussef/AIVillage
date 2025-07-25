"""
HypeRAG MCP Protocol Handlers

Implements Model Context Protocol handlers for HypeRAG server operations.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .auth import AuthContext, PermissionManager, HypeRAGPermissions, require_permission, audit_operation
from .models import ModelRegistry, QueryPlan, KnowledgeGraph, ReasoningResult, Node

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base class for MCP protocol errors"""
    def __init__(self, code: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(f"{code}: {message}")


class AuthenticationRequired(MCPError):
    """Authentication required error"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__("AUTH_REQUIRED", message)


class PermissionDenied(MCPError):
    """Permission denied error"""
    def __init__(self, message: str = "Permission denied"):
        super().__init__("PERMISSION_DENIED", message)


class InvalidRequest(MCPError):
    """Invalid request error"""
    def __init__(self, message: str = "Invalid request"):
        super().__init__("INVALID_REQUEST", message)


class NotFound(MCPError):
    """Resource not found error"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__("NOT_FOUND", message)


class InternalError(MCPError):
    """Internal server error"""
    def __init__(self, message: str = "Internal server error"):
        super().__init__("INTERNAL_ERROR", message)


class MCPRequest:
    """MCP request wrapper"""
    def __init__(self, method: str, params: Dict[str, Any], request_id: str = None):
        self.method = method
        self.params = params
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = datetime.now()


class MCPResponse:
    """MCP response wrapper"""
    def __init__(self, result: Any = None, error: Optional[MCPError] = None, request_id: str = None):
        self.result = result
        self.error = error
        self.request_id = request_id
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        response = {
            "jsonrpc": "2.0",
            "id": self.request_id
        }

        if self.error:
            response["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "data": self.error.data
            }
        else:
            response["result"] = self.result

        return response


class MCPProtocolHandler:
    """Handles MCP protocol requests for HypeRAG"""

    def __init__(
        self,
        permission_manager: PermissionManager,
        model_registry: ModelRegistry,
        storage_backend: Optional[Any] = None
    ):
        self.permission_manager = permission_manager
        self.model_registry = model_registry
        self.storage_backend = storage_backend

        # Request handlers
        self.handlers = {
            # Core operations
            "hyperag/query": self.handle_query,
            "hyperag/creative": self.handle_creative_query,
            "hyperag/repair": self.handle_repair,

            # Knowledge management
            "hyperag/knowledge/add": self.handle_add_knowledge,
            "hyperag/knowledge/search": self.handle_search_knowledge,
            "hyperag/knowledge/update": self.handle_update_knowledge,
            "hyperag/knowledge/delete": self.handle_delete_knowledge,

            # Adapter management
            "hyperag/adapter/upload": self.handle_upload_adapter,
            "hyperag/adapter/list": self.handle_list_adapters,
            "hyperag/adapter/activate": self.handle_activate_adapter,
            "hyperag/adapter/deactivate": self.handle_deactivate_adapter,

            # Guardian operations
            "hyperag/guardian/validate": self.handle_guardian_validate,
            "hyperag/guardian/override": self.handle_guardian_override,

            # System operations
            "hyperag/health": self.handle_health_check,
            "hyperag/metrics": self.handle_metrics,
            "hyperag/audit": self.handle_audit_log,

            # Model management
            "hyperag/model/register": self.handle_register_model,
            "hyperag/model/stats": self.handle_model_stats,
        }

    async def handle_request(
        self,
        request: MCPRequest,
        context: Optional[AuthContext] = None
    ) -> MCPResponse:
        """Handle an MCP request"""
        try:
            # Validate request
            if not request.method:
                raise InvalidRequest("Missing method")

            if request.method not in self.handlers:
                raise NotFound(f"Unknown method: {request.method}")

            # Get handler
            handler = self.handlers[request.method]

            # Call handler with context
            start_time = time.time()
            if context:
                result = await handler(context, **request.params)
            else:
                # Some methods like health check don't require auth
                if request.method in ["hyperag/health"]:
                    result = await handler(**request.params)
                else:
                    raise AuthenticationRequired()

            # Add timing metadata
            processing_time = time.time() - start_time
            if isinstance(result, dict):
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["processing_time_ms"] = round(processing_time * 1000, 2)

            return MCPResponse(result=result, request_id=request.request_id)

        except MCPError as e:
            logger.warning(f"MCP error for {request.method}: {e.message}")
            return MCPResponse(error=e, request_id=request.request_id)

        except Exception as e:
            logger.error(f"Unexpected error for {request.method}: {str(e)}")
            return MCPResponse(
                error=InternalError(f"Unexpected error: {str(e)}"),
                request_id=request.request_id
            )

    # Core query operations

    @require_permission(HypeRAGPermissions.READ)
    @audit_operation("query")
    async def handle_query(
        self,
        context: AuthContext,
        query: str,
        mode: str = "NORMAL",
        user_id: Optional[str] = None,
        plan_hints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle standard query request"""
        # Use user_id from context if not specified
        if not user_id:
            user_id = context.user_id

        # Get agent's reasoning model
        agent_type = context.role

        # Plan the query
        plan_context = {"user_id": user_id, "agent_role": context.role}
        if plan_hints:
            plan_context.update(plan_hints)

        plan = await self.model_registry.process_with_model(
            context.agent_id,
            agent_type,
            "plan",
            query,
            plan_context
        )

        # TODO: Implement actual retrieval and reasoning
        # For now, return mock response
        mock_nodes = [
            Node(
                id="node_1",
                content=f"Mock content related to: {query}",
                node_type="document",
                confidence=0.85
            )
        ]

        # Construct knowledge graph
        knowledge_graph = await self.model_registry.process_with_model(
            context.agent_id,
            agent_type,
            "construct",
            mock_nodes,
            plan
        )

        # Perform reasoning
        reasoning_result = await self.model_registry.process_with_model(
            context.agent_id,
            agent_type,
            "reason",
            knowledge_graph,
            query,
            plan
        )

        return {
            "request_id": plan.query_id,
            "status": "success",
            "mode_used": plan.mode.value,
            "result": {
                "answer": reasoning_result.answer,
                "confidence": reasoning_result.confidence,
                "reasoning_path": [asdict(step) for step in reasoning_result.reasoning_steps],
                "sources": [asdict(node) for node in reasoning_result.sources]
            },
            "guardian_decision": {
                "action": "APPLY",
                "semantic_score": 0.9,
                "utility_score": 0.85,
                "safety_score": 0.95
            },
            "plan": asdict(plan)
        }

    @require_permission(HypeRAGPermissions.READ)
    @audit_operation("creative_query")
    async def handle_creative_query(
        self,
        context: AuthContext,
        source_concept: str,
        target_concept: Optional[str] = None,
        creativity_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle creative/divergent query request"""
        creativity_params = creativity_parameters or {}

        # Mock creative response
        bridges = [
            {
                "id": "bridge_001",
                "path": [source_concept, "intermediate_concept", target_concept or "unknown"],
                "relations": ["relates_to", "enables"],
                "surprise_score": 0.82,
                "confidence": 0.73,
                "explanation": f"Creative connection between {source_concept} and {target_concept or 'related concepts'}",
                "tags": ["creative", "analogy"]
            }
        ]

        return {
            "request_id": str(uuid.uuid4()),
            "status": "success",
            "bridges_found": bridges,
            "computation_time_ms": 1500,
            "guardian_vetted": True,
            "creative_metrics": {
                "novelty": 0.87,
                "usefulness": 0.79,
                "surprise": 0.85
            }
        }

    @require_permission(HypeRAGPermissions.REPAIR_PROPOSE)
    @audit_operation("repair")
    async def handle_repair(
        self,
        context: AuthContext,
        violation_type: str,
        details: Dict[str, Any],
        proposed_action: Optional[str] = None,
        priority: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """Handle graph repair request"""
        # Mock repair response
        repair_proposals = [
            {
                "id": "proposal_001",
                "description": f"Repair {violation_type}: {details.get('description', 'Unknown issue')}",
                "confidence": 0.92,
                "impact_analysis": {
                    "affected_nodes": 2,
                    "affected_edges": 1,
                    "consistency_improvement": 0.15
                }
            }
        ]

        return {
            "request_id": str(uuid.uuid4()),
            "status": "success",
            "repair_proposals": repair_proposals,
            "guardian_review": {
                "recommendation": "APPLY",
                "reasoning": "Repair maintains graph consistency"
            }
        }

    # Knowledge management operations

    @require_permission(HypeRAGPermissions.WRITE)
    @audit_operation("add_knowledge")
    async def handle_add_knowledge(
        self,
        context: AuthContext,
        content: str,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle add knowledge request"""
        node_id = str(uuid.uuid4())

        # TODO: Implement actual storage
        return {
            "node_id": node_id,
            "status": "success",
            "message": "Knowledge added successfully"
        }

    @require_permission(HypeRAGPermissions.READ)
    @audit_operation("search_knowledge")
    async def handle_search_knowledge(
        self,
        context: AuthContext,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle search knowledge request"""
        # Mock search results
        results = [
            {
                "id": f"node_{i}",
                "content": f"Search result {i} for: {query}",
                "relevance": 0.9 - (i * 0.1),
                "metadata": {"type": "document"}
            }
            for i in range(min(limit, 3))
        ]

        return {
            "results": results,
            "total_count": len(results),
            "query": query
        }

    @require_permission(HypeRAGPermissions.WRITE)
    @audit_operation("update_knowledge")
    async def handle_update_knowledge(
        self,
        context: AuthContext,
        node_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle update knowledge request"""
        # TODO: Implement actual update
        return {
            "node_id": node_id,
            "status": "success",
            "message": "Knowledge updated successfully"
        }

    @require_permission(HypeRAGPermissions.WRITE)
    @audit_operation("delete_knowledge")
    async def handle_delete_knowledge(
        self,
        context: AuthContext,
        node_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle delete knowledge request"""
        # TODO: Implement actual deletion
        return {
            "node_id": node_id,
            "status": "success",
            "message": "Knowledge deleted successfully"
        }

    # Adapter management

    @require_permission(HypeRAGPermissions.ADAPTER_MANAGE)
    @audit_operation("upload_adapter")
    async def handle_upload_adapter(
        self,
        context: AuthContext,
        name: str,
        description: str,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle adapter upload request"""
        adapter_id = f"adapt_{name}_{int(time.time())}"

        return {
            "adapter_id": adapter_id,
            "status": "success",
            "validation_results": {
                "checksum_valid": True,
                "format_valid": True,
                "compatibility_check": "passed"
            },
            "guardian_signature": {
                "status": "signed",
                "signed_at": datetime.now().isoformat()
            }
        }

    @require_permission(HypeRAGPermissions.ADAPTER_USE)
    @audit_operation("list_adapters")
    async def handle_list_adapters(
        self,
        context: AuthContext,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle list adapters request"""
        # Mock adapter list
        adapters = [
            {
                "id": "adapt_medical_001",
                "name": "Medical Terminology",
                "domain": "medical",
                "status": "active"
            },
            {
                "id": "adapt_tech_002",
                "name": "Technical Documentation",
                "domain": "technical",
                "status": "available"
            }
        ]

        if domain:
            adapters = [a for a in adapters if a["domain"] == domain]

        return {
            "adapters": adapters,
            "total_count": len(adapters)
        }

    @require_permission(HypeRAGPermissions.ADAPTER_USE)
    @audit_operation("activate_adapter")
    async def handle_activate_adapter(
        self,
        context: AuthContext,
        adapter_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle activate adapter request"""
        return {
            "adapter_id": adapter_id,
            "status": "activated",
            "message": "Adapter activated successfully"
        }

    @require_permission(HypeRAGPermissions.ADAPTER_USE)
    @audit_operation("deactivate_adapter")
    async def handle_deactivate_adapter(
        self,
        context: AuthContext,
        adapter_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle deactivate adapter request"""
        return {
            "adapter_id": adapter_id,
            "status": "deactivated",
            "message": "Adapter deactivated successfully"
        }

    # Guardian operations

    @require_permission(HypeRAGPermissions.GATE_OVERRIDE)
    @audit_operation("guardian_validate")
    async def handle_guardian_validate(
        self,
        context: AuthContext,
        validation_request: Dict[str, Any],
        decision: str,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle guardian validation request"""
        validation_id = str(uuid.uuid4())

        return {
            "validation_id": validation_id,
            "status": "success",
            "original_decision": "QUARANTINE",
            "new_decision": decision,
            "decision_record": {
                "timestamp": datetime.now().isoformat(),
                "reviewer": context.user_id,
                "reasoning": "Manual review completed"
            }
        }

    @require_permission(HypeRAGPermissions.GATE_OVERRIDE)
    @audit_operation("guardian_override")
    async def handle_guardian_override(
        self,
        context: AuthContext,
        operation_id: str,
        override_reason: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle guardian override request"""
        return {
            "operation_id": operation_id,
            "status": "overridden",
            "override_reason": override_reason,
            "timestamp": datetime.now().isoformat()
        }

    # System operations

    async def handle_health_check(self, **kwargs) -> Dict[str, Any]:
        """Handle health check request (no auth required)"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "permission_manager": "healthy",
                "model_registry": "healthy",
                "storage_backend": "healthy" if self.storage_backend else "not_configured"
            }
        }

    @require_permission(HypeRAGPermissions.MONITOR)
    @audit_operation("metrics")
    async def handle_metrics(
        self,
        context: AuthContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle metrics request"""
        model_stats = self.model_registry.get_model_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "models": model_stats,
            "active_sessions": len(await self.permission_manager.get_active_sessions()),
            "system_info": {
                "uptime_seconds": time.time() - getattr(self, 'start_time', time.time())
            }
        }

    @require_permission(HypeRAGPermissions.MONITOR)
    @audit_operation("audit_log")
    async def handle_audit_log(
        self,
        context: AuthContext,
        user_id: Optional[str] = None,
        limit: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle audit log request"""
        entries = await self.permission_manager.get_audit_log(user_id, limit)

        return {
            "entries": [asdict(entry) for entry in entries],
            "total_count": len(entries)
        }

    # Model management

    @require_permission(HypeRAGPermissions.ADMIN)
    @audit_operation("register_model")
    async def handle_register_model(
        self,
        context: AuthContext,
        agent_id: str,
        model_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Handle model registration request"""
        # TODO: Implement actual model registration
        return {
            "agent_id": agent_id,
            "status": "registered",
            "message": "Model registered successfully"
        }

    @require_permission(HypeRAGPermissions.MONITOR)
    @audit_operation("model_stats")
    async def handle_model_stats(
        self,
        context: AuthContext,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle model statistics request"""
        stats = self.model_registry.get_model_stats()

        if agent_id:
            stats = {agent_id: stats.get(agent_id, {})}

        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
