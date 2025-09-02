"""
RBAC API Server for AIVillage.

Provides REST API endpoints for role-based access control, multi-tenant isolation,
and secure access to all AIVillage systems.
"""

import asyncio
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from typing import Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .aivillage_rbac_integration import AIVillageRBACIntegration, initialize_aivillage_rbac
from .rbac_system import Role

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:

    class CreateTenantRequest(BaseModel):
        name: str
        admin_username: str
        admin_email: str
        admin_password: str
        config: dict[str, Any] | None = None

    class CreateUserRequest(BaseModel):
        username: str
        email: str
        password: str
        tenant_id: str
        role: str = "user"

    class AuthRequest(BaseModel):
        username: str
        password: str
        tenant_id: str

    class AgentRequest(BaseModel):
        name: str
        type: str
        config: dict[str, Any] | None = None

    class RAGCollectionRequest(BaseModel):
        name: str
        config: dict[str, Any] | None = None

    class RAGQueryRequest(BaseModel):
        query: str
        max_results: int = 10

    class P2PNetworkRequest(BaseModel):
        name: str
        config: dict[str, Any] | None = None

    class APIResponse(BaseModel):
        status: int
        data: dict[str, Any] | None = None
        error: str | None = None

    # Security dependency
    security = HTTPBearer()


class RBACAPIServer:
    """FastAPI-based RBAC API server."""

    def __init__(self, integration: AIVillageRBACIntegration):
        self.integration = integration
        self.app = None
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="AIVillage RBAC API",
                description="Role-Based Access Control and Multi-Tenant API for AIVillage",
                version="1.0.0",
            )
            self._setup_fastapi_routes()
        else:
            logger.warning("FastAPI not available, using basic HTTP server")

    def _setup_fastapi_routes(self):
        """Set up FastAPI routes."""
        if not self.app:
            return

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Authentication dependency
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            token = credentials.credentials
            payload = await self.integration.rbac.verify_token(token)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid token")
            return payload

        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(request: AuthRequest, client_request: Request):
            try:
                session = await self.integration.rbac.authenticate(
                    username=request.username,
                    password=request.password,
                    tenant_id=request.tenant_id,
                    ip_address=client_request.client.host,
                    user_agent=client_request.headers.get("user-agent", ""),
                )

                return {
                    "access_token": session.token,
                    "refresh_token": session.refresh_token,
                    "token_type": "bearer",
                    "expires_at": session.expires_at.isoformat(),
                    "user_id": session.user_id,
                    "tenant_id": session.tenant_id,
                }
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))

        @self.app.post("/auth/refresh")
        async def refresh_token(refresh_token: str):
            try:
                session = await self.integration.rbac.refresh_token(refresh_token)
                if not session:
                    raise HTTPException(status_code=401, detail="Invalid refresh token")

                return {
                    "access_token": session.token,
                    "refresh_token": session.refresh_token,
                    "token_type": "bearer",
                    "expires_at": session.expires_at.isoformat(),
                }
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))

        @self.app.post("/auth/logout")
        async def logout(current_user: dict = Depends(get_current_user)):
            await self.integration.rbac.revoke_session(current_user["session_id"])
            return {"message": "Logged out successfully"}

        # Tenant management
        @self.app.post("/tenants")
        async def create_tenant(request: CreateTenantRequest):
            try:
                tenant = await self.integration.rbac.create_tenant(
                    name=request.name,
                    admin_user={
                        "username": request.admin_username,
                        "email": request.admin_email,
                        "password": request.admin_password,
                    },
                    config=request.config,
                )

                return {"tenant_id": tenant.tenant_id, "name": tenant.name, "created_at": tenant.created_at.isoformat()}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/tenants/{tenant_id}/usage")
        async def get_tenant_usage(tenant_id: str, current_user: dict = Depends(get_current_user)):
            try:
                usage = await self.integration.tenant_manager.get_tenant_usage(
                    tenant_id=tenant_id, user_id=current_user["user_id"]
                )
                return usage
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e))

        # User management
        @self.app.post("/users")
        async def create_user(request: CreateUserRequest, current_user: dict = Depends(get_current_user)):
            try:
                # Check if user can create users in this tenant
                if current_user["tenant_id"] != request.tenant_id:
                    user = self.integration.rbac.users.get(current_user["user_id"])
                    if not user or user.role.value != "super_admin":
                        raise HTTPException(status_code=403, detail="Cannot create users in other tenants")

                role = Role(request.role)
                user = await self.integration.rbac.create_user(
                    username=request.username,
                    email=request.email,
                    password=request.password,
                    tenant_id=request.tenant_id,
                    role=role,
                )

                return {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "tenant_id": user.tenant_id,
                    "created_at": user.created_at.isoformat(),
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Agent endpoints
        @self.app.post("/agents")
        async def create_agent(request: AgentRequest, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agents",
                action="create_agent",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                params=request.dict(),
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/agents")
        async def list_agents(current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agents",
                action="list_agents",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agents",
                action="get_agent",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=agent_id,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.post("/agents/{agent_id}/execute")
        async def execute_agent(agent_id: str, params: dict[str, Any], current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agents",
                action="execute_agent",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=agent_id,
                params=params,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.delete("/agents/{agent_id}")
        async def delete_agent(agent_id: str, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agents",
                action="delete_agent",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=agent_id,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        # RAG endpoints
        @self.app.post("/rag/collections")
        async def create_rag_collection(request: RAGCollectionRequest, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="rag",
                action="create_collection",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                params=request.dict(),
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/rag/collections")
        async def list_rag_collections(current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="rag",
                action="list_collections",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.post("/rag/collections/{collection_id}/query")
        async def query_rag(
            collection_id: str, request: RAGQueryRequest, current_user: dict = Depends(get_current_user)
        ):
            result = await self.integration.secure_api_call(
                system="rag",
                action="query_rag",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=collection_id,
                params=request.dict(),
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.post("/rag/collections/{collection_id}/documents")
        async def add_documents(
            collection_id: str, documents: dict[str, Any], current_user: dict = Depends(get_current_user)
        ):
            result = await self.integration.secure_api_call(
                system="rag",
                action="add_documents",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=collection_id,
                params=documents,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        # P2P Network endpoints
        @self.app.post("/p2p/networks")
        async def create_p2p_network(request: P2PNetworkRequest, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="p2p",
                action="create_network",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                params=request.dict(),
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/p2p/networks")
        async def list_p2p_networks(current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="p2p",
                action="list_networks",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.post("/p2p/networks/{network_id}/join")
        async def join_p2p_network(
            network_id: str, params: dict[str, Any], current_user: dict = Depends(get_current_user)
        ):
            result = await self.integration.secure_api_call(
                system="p2p",
                action="join_network",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=network_id,
                params=params,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        # Agent Forge endpoints
        @self.app.post("/agent_forge/training")
        async def start_agent_forge_training(params: dict[str, Any], current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agent_forge",
                action="start_training",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                params=params,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/agent_forge/training/{job_id}")
        async def get_training_status(job_id: str, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="agent_forge",
                action="training_status",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=job_id,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        # Digital Twin endpoints
        @self.app.post("/digital-twin")
        async def create_digital_twin(params: dict[str, Any], current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="digital_twin",
                action="create_twin",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                params=params,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/digital-twin/{twin_id}/privacy")
        async def get_privacy_report(twin_id: str, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="digital_twin",
                action="privacy_report",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=twin_id,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        # Mobile endpoints
        @self.app.post("/mobile/devices")
        async def register_mobile_device(params: dict[str, Any], current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="mobile",
                action="register_device",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                params=params,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        @self.app.get("/mobile/devices/{device_id}")
        async def get_device_status(device_id: str, current_user: dict = Depends(get_current_user)):
            result = await self.integration.secure_api_call(
                system="mobile",
                action="device_status",
                user_id=current_user["user_id"],
                tenant_id=current_user["tenant_id"],
                resource_id=device_id,
            )

            if result.get("status", 500) >= 400:
                raise HTTPException(status_code=result["status"], detail=result.get("error"))

            return result["data"]

        # Audit endpoints
        @self.app.get("/audit/logs")
        async def get_audit_logs(
            start_date: str | None = None,
            end_date: str | None = None,
            action_filter: str | None = None,
            current_user: dict = Depends(get_current_user),
        ):
            try:
                start = datetime.fromisoformat(start_date) if start_date else None
                end = datetime.fromisoformat(end_date) if end_date else None

                logs = await self.integration.rbac.get_audit_log(
                    user_id=current_user["user_id"], start_date=start, end_date=end, action_filter=action_filter
                )

                return {"logs": logs}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API server."""
        if FASTAPI_AVAILABLE and self.app:
            logger.info(f"Starting RBAC API server on {host}:{port}")
            config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        else:
            logger.error("FastAPI not available, cannot start server")


class BasicHTTPHandler(BaseHTTPRequestHandler):
    """Basic HTTP handler for fallback when FastAPI is not available."""

    def __init__(self, integration: AIVillageRBACIntegration, *args, **kwargs):
        self.integration = integration
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = json.dumps(
                {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Basic HTTP server running (install FastAPI for full API)",
                }
            )
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        self.send_response(501)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps(
            {"error": "Full API requires FastAPI installation", "message": "pip install fastapi uvicorn"}
        )
        self.wfile.write(response.encode())


async def create_rbac_server() -> RBACAPIServer:
    """Create and configure RBAC API server."""
    # Initialize RBAC integration
    integration = await initialize_aivillage_rbac()

    # Create server
    server = RBACAPIServer(integration)

    logger.info("RBAC API server created successfully")
    return server


if __name__ == "__main__":

    async def main():
        # Create and start server
        server = await create_rbac_server()

        if FASTAPI_AVAILABLE:
            await server.start_server()
        else:
            # Fallback to basic HTTP server
            integration = await initialize_aivillage_rbac()

            def handler_factory(*args, **kwargs):
                return BasicHTTPHandler(integration, *args, **kwargs)

            httpd = HTTPServer(("localhost", 8000), handler_factory)
            logger.info("Starting basic HTTP server on localhost:8000")
            logger.info("Install FastAPI for full functionality: pip install fastapi uvicorn")
            httpd.serve_forever()

    asyncio.run(main())
