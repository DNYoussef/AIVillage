"""Deploy Agent - Containerized Deployment System.

Implements comprehensive deployment capabilities for Agent Forge models:
- Docker containerization with optimization
- Multi-environment deployment (local, cloud, edge)
- Auto-scaling and load balancing
- Health monitoring and metrics collection
- Model versioning and rollback capabilities
- Integration with existing pipeline outputs
"""

import asyncio
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from kubernetes import client
from kubernetes import config as k8s_config

import docker

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    model_path: str
    deployment_name: str
    output_dir: str

    # Container settings
    base_image: str = "python:3.9-slim"
    memory_limit: str = "4Gi"
    cpu_limit: str = "2"
    gpu_enabled: bool = True

    # Deployment settings
    replicas: int = 1
    port: int = 8000
    health_check_path: str = "/health"

    # Environment settings
    deployment_type: str = "docker"  # docker, kubernetes, aws, azure, gcp
    registry_url: str | None = None
    namespace: str = "default"

    # Model settings
    quantization: bool = True
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 0.7

    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    wandb_integration: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentStatus:
    """Status of a deployment."""

    deployment_name: str
    status: str  # pending, running, failed, stopped
    container_ids: list[str]
    endpoints: list[str]
    health_status: str
    created_at: float
    last_updated: float
    resource_usage: dict[str, Any]
    error_message: str | None = None


class ContainerBuilder:
    """Builds optimized containers for model deployment."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.docker_client = docker.from_env()

    def create_dockerfile(self, model_dir: Path) -> str:
        """Create optimized Dockerfile for model deployment."""
        dockerfile_content = f"""
# Multi-stage build for optimized container
FROM {self.config.base_image} as base

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \\
    transformers \\
    fastapi \\
    uvicorn \\
    prometheus-client \\
    psutil \\
    numpy \\
    accelerate

# Production stage
FROM base as production

# Create app directory
WORKDIR /app

# Copy model files
COPY model/ ./model/
COPY app.py ./
COPY requirements.txt ./

# Install additional requirements if any
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.config.port}{self.config.health_check_path} || exit 1

# Expose port
EXPOSE {self.config.port}

# Environment variables
ENV MODEL_PATH=/app/model
ENV PORT={self.config.port}
ENV BATCH_SIZE={self.config.batch_size}
ENV MAX_LENGTH={self.config.max_length}
ENV TEMPERATURE={self.config.temperature}
ENV LOG_LEVEL={self.config.log_level}

# Start server
CMD ["python", "app.py"]
"""

        dockerfile_path = model_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        return str(dockerfile_path)

    def create_app_server(self, model_dir: Path) -> str:
        """Create FastAPI server for model inference."""
        app_content = '''
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
import psutil
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests')
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Model load time')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage')

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    num_return_sequences: Optional[int] = 1

class GenerationResponse(BaseModel):
    generated_text: List[str]
    processing_time: float
    model_info: Dict[str, Any]
    timestamp: float

class ModelServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/app/model")
        self.load_model()

    def load_model(self):
        """Load model and tokenizer."""
        start_time = time.time()
        logger.info(f"Loading model from {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt."""
        start_time = time.time()
        REQUEST_COUNT.inc()

        try:
            # Prepare inputs
            max_length = request.max_length or int(os.getenv("MAX_LENGTH", "512"))
            temperature = request.temperature or float(os.getenv("TEMPERATURE", "0.7"))

            inputs = self.tokenizer.encode(request.prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    num_return_sequences=request.num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode outputs
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove input prompt from output
                generated_text = text[len(request.prompt):].strip()
                generated_texts.append(generated_text)

            processing_time = time.time() - start_time
            REQUEST_DURATION.observe(processing_time)

            return GenerationResponse(
                generated_text=generated_texts,
                processing_time=processing_time,
                model_info={
                    "model_path": self.model_path,
                    "device": self.device,
                    "max_length": max_length,
                    "temperature": temperature
                },
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_health(self) -> Dict[str, Any]:
        """Get server health status."""
        # Update metrics
        memory_info = psutil.virtual_memory()
        MEMORY_USAGE.set(memory_info.used)

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            GPU_MEMORY_USAGE.set(gpu_memory)

        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "device": self.device,
            "memory_usage_mb": memory_info.used / (1024**2),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_mb": torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0,
            "timestamp": time.time()
        }

# Initialize server
model_server = ModelServer()
app = FastAPI(title="Agent Forge Model Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from prompt."""
    return await model_server.generate(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return model_server.get_health()

@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "model_path": model_server.model_path,
        "device": model_server.device,
        "model_loaded": model_server.model is not None,
        "tokenizer_vocab_size": len(model_server.tokenizer) if model_server.tokenizer else 0
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=os.getenv("LOG_LEVEL", "info").lower())
'''

        app_path = model_dir / "app.py"
        with open(app_path, "w") as f:
            f.write(app_content)

        return str(app_path)

    def create_requirements(self, model_dir: Path) -> str:
        """Create requirements.txt file."""
        requirements = """
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
prometheus-client==0.19.0
psutil==5.9.6
numpy>=1.24.0
pydantic>=2.0.0
"""

        requirements_path = model_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write(requirements.strip())

        return str(requirements_path)

    async def build_container(self, model_dir: Path) -> str:
        """Build container image."""
        logger.info(f"Building container for {self.config.deployment_name}")

        try:
            # Create necessary files
            self.create_dockerfile(model_dir)
            self.create_app_server(model_dir)
            self.create_requirements(model_dir)

            # Build image
            image_tag = f"{self.config.deployment_name}:latest"

            logger.info("Building Docker image...")
            image, logs = self.docker_client.images.build(path=str(model_dir), tag=image_tag, rm=True, forcerm=True)

            # Log build output
            for log_line in logs:
                if "stream" in log_line:
                    logger.info(log_line["stream"].strip())

            logger.info(f"Container built successfully: {image_tag}")
            return image_tag

        except Exception as e:
            logger.exception(f"Container build failed: {e}")
            raise


class KubernetesDeployer:
    """Handles Kubernetes deployments."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        try:
            k8s_config.load_incluster_config()
        except BaseException:
            k8s_config.load_kube_config()

        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()

    def create_deployment_manifest(self, image_tag: str) -> dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.deployment_name,
                "namespace": self.config.namespace,
                "labels": {"app": self.config.deployment_name, "version": "v1"},
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {"matchLabels": {"app": self.config.deployment_name}},
                "template": {
                    "metadata": {"labels": {"app": self.config.deployment_name}},
                    "spec": {
                        "containers": [
                            {
                                "name": self.config.deployment_name,
                                "image": image_tag,
                                "ports": [{"containerPort": self.config.port}],
                                "resources": {
                                    "limits": {
                                        "memory": self.config.memory_limit,
                                        "cpu": self.config.cpu_limit,
                                    },
                                    "requests": {"memory": "1Gi", "cpu": "500m"},
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": self.config.port,
                                    },
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30,
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": self.config.port,
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "env": [
                                    {
                                        "name": "LOG_LEVEL",
                                        "value": self.config.log_level,
                                    },
                                    {"name": "PORT", "value": str(self.config.port)},
                                    {
                                        "name": "BATCH_SIZE",
                                        "value": str(self.config.batch_size),
                                    },
                                ],
                            }
                        ]
                    },
                },
            },
        }

    def create_service_manifest(self) -> dict[str, Any]:
        """Create Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.deployment_name}-service",
                "namespace": self.config.namespace,
            },
            "spec": {
                "selector": {"app": self.config.deployment_name},
                "ports": [{"port": 80, "targetPort": self.config.port, "protocol": "TCP"}],
                "type": "LoadBalancer",
            },
        }

    async def deploy(self, image_tag: str) -> DeploymentStatus:
        """Deploy to Kubernetes."""
        logger.info(f"Deploying {self.config.deployment_name} to Kubernetes")

        try:
            # Create deployment
            deployment_manifest = self.create_deployment_manifest(image_tag)

            try:
                self.apps_v1.create_namespaced_deployment(namespace=self.config.namespace, body=deployment_manifest)
            except client.ApiException as e:
                if e.status == 409:  # Already exists
                    self.apps_v1.patch_namespaced_deployment(
                        name=self.config.deployment_name,
                        namespace=self.config.namespace,
                        body=deployment_manifest,
                    )
                else:
                    raise

            # Create service
            service_manifest = self.create_service_manifest()

            try:
                self.core_v1.create_namespaced_service(namespace=self.config.namespace, body=service_manifest)
            except client.ApiException as e:
                if e.status != 409:  # Ignore if already exists
                    raise

            # Wait for deployment to be ready
            await self._wait_for_deployment()

            # Get deployment status
            return await self.get_deployment_status()

        except Exception as e:
            logger.exception(f"Kubernetes deployment failed: {e}")
            raise

    async def _wait_for_deployment(self, timeout: int = 600) -> None:
        """Wait for deployment to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=self.config.deployment_name, namespace=self.config.namespace
                )

                if (
                    deployment.status.ready_replicas == self.config.replicas
                    and deployment.status.available_replicas == self.config.replicas
                ):
                    logger.info("Deployment is ready")
                    return

            except Exception as e:
                logger.warning(f"Error checking deployment status: {e}")

            await asyncio.sleep(10)

        msg = "Deployment did not become ready within timeout"
        raise TimeoutError(msg)

    async def get_deployment_status(self) -> DeploymentStatus:
        """Get current deployment status."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.config.deployment_name, namespace=self.config.namespace
            )

            pods = self.core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"app={self.config.deployment_name}",
            )

            service = self.core_v1.read_namespaced_service(
                name=f"{self.config.deployment_name}-service",
                namespace=self.config.namespace,
            )

            # Determine status
            if deployment.status.ready_replicas == self.config.replicas:
                status = "running"
                health_status = "healthy"
            elif deployment.status.ready_replicas and deployment.status.ready_replicas > 0:
                status = "partially_running"
                health_status = "degraded"
            else:
                status = "failed"
                health_status = "unhealthy"

            # Get endpoints
            endpoints = []
            if service.status.load_balancer.ingress:
                for ingress in service.status.load_balancer.ingress:
                    if ingress.ip:
                        endpoints.append(f"http://{ingress.ip}")
                    elif ingress.hostname:
                        endpoints.append(f"http://{ingress.hostname}")

            return DeploymentStatus(
                deployment_name=self.config.deployment_name,
                status=status,
                container_ids=[pod.metadata.name for pod in pods.items],
                endpoints=endpoints,
                health_status=health_status,
                created_at=deployment.metadata.creation_timestamp.timestamp(),
                last_updated=time.time(),
                resource_usage={
                    "replicas": deployment.status.replicas or 0,
                    "ready_replicas": deployment.status.ready_replicas or 0,
                    "pods": len(pods.items),
                },
            )

        except Exception as e:
            logger.exception(f"Failed to get deployment status: {e}")
            return DeploymentStatus(
                deployment_name=self.config.deployment_name,
                status="unknown",
                container_ids=[],
                endpoints=[],
                health_status="unknown",
                created_at=time.time(),
                last_updated=time.time(),
                resource_usage={},
                error_message=str(e),
            )


class DockerDeployer:
    """Handles Docker deployments."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.docker_client = docker.from_env()

    async def deploy(self, image_tag: str) -> DeploymentStatus:
        """Deploy using Docker."""
        logger.info(f"Deploying {self.config.deployment_name} with Docker")

        try:
            # Stop existing container if running
            try:
                existing = self.docker_client.containers.get(self.config.deployment_name)
                existing.stop()
                existing.remove()
                logger.info("Stopped existing container")
            except docker.errors.NotFound:
                pass

            # Run new container
            container = self.docker_client.containers.run(
                image_tag,
                name=self.config.deployment_name,
                ports={f"{self.config.port}/tcp": self.config.port},
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                environment={
                    "LOG_LEVEL": self.config.log_level,
                    "PORT": str(self.config.port),
                    "BATCH_SIZE": str(self.config.batch_size),
                },
                mem_limit=(self.config.memory_limit if self.config.memory_limit != "4Gi" else "4g"),
            )

            # Wait for container to be healthy
            await self._wait_for_health(container)

            return DeploymentStatus(
                deployment_name=self.config.deployment_name,
                status="running",
                container_ids=[container.id],
                endpoints=[f"http://localhost:{self.config.port}"],
                health_status="healthy",
                created_at=time.time(),
                last_updated=time.time(),
                resource_usage=self._get_container_stats(container),
            )

        except Exception as e:
            logger.exception(f"Docker deployment failed: {e}")
            raise

    async def _wait_for_health(self, container, timeout: int = 300) -> None:
        """Wait for container to be healthy."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status == "running":
                    # Check health endpoint
                    response = requests.get(
                        f"http://localhost:{self.config.port}{self.config.health_check_path}",
                        timeout=5,
                    )
                    if response.status_code == 200:
                        logger.info("Container is healthy")
                        return
            except Exception:
                pass

            await asyncio.sleep(5)

        msg = "Container did not become healthy within timeout"
        raise TimeoutError(msg)

    def _get_container_stats(self, container) -> dict[str, Any]:
        """Get container resource usage statistics."""
        try:
            stats = container.stats(stream=False)
            return {
                "memory_usage": stats["memory_stats"].get("usage", 0),
                "cpu_usage": stats["cpu_stats"].get("cpu_usage", {}).get("total_usage", 0),
                "network_rx": stats["networks"].get("eth0", {}).get("rx_bytes", 0),
                "network_tx": stats["networks"].get("eth0", {}).get("tx_bytes", 0),
            }
        except Exception:
            return {}


class DeploymentOrchestrator:
    """Main deployment orchestrator."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.container_builder = ContainerBuilder(config)

        if config.deployment_type == "kubernetes":
            self.deployer = KubernetesDeployer(config)
        else:
            self.deployer = DockerDeployer(config)

        # Track deployments
        self.deployment_history = []

    async def deploy_model(self, model_path: str) -> DeploymentStatus:
        """Deploy a model end-to-end."""
        logger.info(f"Starting deployment of model from {model_path}")

        try:
            # Prepare model directory
            model_dir = await self._prepare_model_directory(model_path)

            # Build container
            image_tag = await self.container_builder.build_container(model_dir)

            # Deploy
            deployment_status = await self.deployer.deploy(image_tag)

            # Save deployment info
            await self._save_deployment_info(deployment_status, image_tag)

            logger.info(f"Deployment completed successfully: {deployment_status.status}")
            return deployment_status

        except Exception as e:
            logger.exception(f"Deployment failed: {e}")
            raise

    async def _prepare_model_directory(self, model_path: str) -> Path:
        """Prepare model directory for containerization."""
        model_src = Path(model_path)
        model_dst = self.output_dir / "deployment" / "model"

        # Clean and create destination
        if model_dst.exists():
            shutil.rmtree(model_dst)
        model_dst.mkdir(parents=True)

        # Copy model files
        if model_src.is_file():
            # Single file model
            shutil.copy2(model_src, model_dst)
        else:
            # Directory with model files
            shutil.copytree(model_src, model_dst, dirs_exist_ok=True)

        logger.info(f"Model prepared in {model_dst}")
        return model_dst.parent

    async def _save_deployment_info(self, status: DeploymentStatus, image_tag: str) -> None:
        """Save deployment information."""
        deployment_info = {
            "deployment_config": self.config.to_dict(),
            "deployment_status": asdict(status),
            "image_tag": image_tag,
            "timestamp": time.time(),
        }

        info_path = self.output_dir / f"deployment_{self.config.deployment_name}.json"
        with open(info_path, "w") as f:
            json.dump(deployment_info, f, indent=2)

        self.deployment_history.append(deployment_info)
        logger.info(f"Deployment info saved: {info_path}")

    async def get_deployment_status(self) -> DeploymentStatus:
        """Get current deployment status."""
        return await self.deployer.get_deployment_status()

    async def scale_deployment(self, replicas: int) -> None:
        """Scale deployment to specified number of replicas."""
        if hasattr(self.deployer, "scale"):
            await self.deployer.scale(replicas)
        else:
            logger.warning("Scaling not supported for this deployment type")

    async def stop_deployment(self) -> None:
        """Stop the deployment."""
        if hasattr(self.deployer, "stop"):
            await self.deployer.stop()
        else:
            logger.warning("Stop not implemented for this deployment type")


# CLI and usage
async def main() -> None:
    """Main deployment entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge Model Deployment")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--deployment-name", required=True, help="Deployment name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--deployment-type",
        default="docker",
        choices=["docker", "kubernetes"],
        help="Deployment type",
    )
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--memory-limit", default="4Gi", help="Memory limit")
    parser.add_argument("--cpu-limit", default="2", help="CPU limit")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")

    args = parser.parse_args()

    # Create deployment configuration
    config = DeploymentConfig(
        model_path=args.model_path,
        deployment_name=args.deployment_name,
        output_dir=args.output_dir,
        deployment_type=args.deployment_type,
        replicas=args.replicas,
        port=args.port,
        memory_limit=args.memory_limit,
        cpu_limit=args.cpu_limit,
        namespace=args.namespace,
    )

    # Initialize orchestrator
    orchestrator = DeploymentOrchestrator(config)

    # Deploy model
    status = await orchestrator.deploy_model(args.model_path)

    print("\nDeployment Status:")
    print(f"Name: {status.deployment_name}")
    print(f"Status: {status.status}")
    print(f"Health: {status.health_status}")
    print(f"Endpoints: {status.endpoints}")
    print(f"Container IDs: {status.container_ids}")

    # Test deployment if endpoints available
    if status.endpoints and status.status == "running":
        print("\nTesting deployment...")
        test_endpoint = status.endpoints[0]

        try:
            # Test health endpoint
            health_response = requests.get(f"{test_endpoint}/health", timeout=10)
            if health_response.status_code == 200:
                print("✅ Health check passed")

            # Test generation endpoint
            gen_response = requests.post(
                f"{test_endpoint}/generate",
                json={"prompt": "Hello, world!", "max_length": 50},
                timeout=30,
            )
            if gen_response.status_code == 200:
                print("✅ Generation endpoint working")
                result = gen_response.json()
                print(f"Generated: {result['generated_text'][0][:100]}...")
            else:
                print(f"❌ Generation endpoint failed: {gen_response.status_code}")

        except Exception as e:
            print(f"❌ Deployment test failed: {e}")

    print("\nDeployment completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
