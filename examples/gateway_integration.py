"""
Example: Integrating Fog API with AIVillage Gateway

This example shows how to mount the fog computing endpoints
onto the existing AIVillage Gateway server running on port 8000.
"""

from fastapi import FastAPI
from packages.fog.gateway import create_admin_api, create_jobs_api, create_sandbox_api, create_usage_api


def integrate_fog_apis(app: FastAPI) -> None:
    """
    Integrate fog computing APIs with existing Gateway server

    Args:
        app: FastAPI application instance (the existing Gateway)
    """

    # Create fog API instances
    jobs_api = create_jobs_api()
    sandbox_api = create_sandbox_api()
    usage_api = create_usage_api()
    admin_api = create_admin_api()

    # Mount fog endpoints on existing Gateway
    app.include_router(jobs_api.router)
    app.include_router(sandbox_api.router)
    app.include_router(usage_api.router)
    app.include_router(admin_api.router)

    print("✅ Fog computing APIs integrated:")
    print("  - POST /v1/fog/jobs - Submit fog jobs")
    print("  - POST /v1/fog/sandboxes - Create sandboxes")
    print("  - GET /v1/fog/usage - Usage tracking")
    print("  - POST /v1/fog/admin/nodes - Node registration")


def create_standalone_fog_server() -> FastAPI:
    """
    Create standalone fog server (for development/testing)

    Returns:
        FastAPI app with fog endpoints
    """

    app = FastAPI(
        title="AIVillage Fog Computing",
        description="Distributed fog computing over BetaNet",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Integrate fog APIs
    integrate_fog_apis(app)

    return app


if __name__ == "__main__":
    import uvicorn

    # Create standalone fog server for testing
    app = create_standalone_fog_server()

    print("🌫️ Starting AIVillage Fog Gateway on http://localhost:8001")
    print("📖 API docs available at http://localhost:8001/docs")

    uvicorn.run(app, host="0.0.0.0", port=8001)
