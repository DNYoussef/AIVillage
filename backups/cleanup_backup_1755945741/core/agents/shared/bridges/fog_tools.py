"""
Agent MCP Fog Tools

MCP (Model Control Protocol) tools for agents to interact with fog computing infrastructure:
- create_sandbox: Create isolated execution environments
- run_job: Submit jobs to fog gateway with namespace validation
- stream_logs: Real-time log streaming from remote jobs
- fetch_artifacts: Download results and outputs

These tools integrate with the existing agent MCP framework and provide secure,
namespace-isolated access to fog computing resources.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import aiohttp

# Import base MCP tool from agent template
from packages.agents.core.base_agent_template import MCPTool

logger = logging.getLogger(__name__)


class CreateSandboxTool(MCPTool):
    """MCP tool for creating isolated execution environments in fog network"""

    def __init__(self, fog_gateway_url: str = "http://localhost:8080"):
        super().__init__("create_sandbox", "Create isolated execution environment in fog network")
        self.fog_gateway_url = fog_gateway_url.rstrip("/")

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Create a new sandbox environment

        Args:
            parameters:
                namespace: str - Namespace for RBAC and quota enforcement
                runtime: str - Execution runtime (wasi, microvm, oci)
                resources: dict - Resource requirements (cpu_cores, memory_gb, etc.)
                name: str - Optional human-readable name

        Returns:
            dict: {
                "status": "success" | "error",
                "sandbox_id": str,
                "endpoint": str,
                "namespace": str,
                "runtime": str,
                "message": str
            }
        """
        self.log_usage()

        # Validate required parameters
        namespace = parameters.get("namespace")
        if not namespace:
            return {"status": "error", "message": "namespace parameter is required", "sandbox_id": None}

        runtime = parameters.get("runtime", "wasi")
        resources = parameters.get("resources", {})
        name = parameters.get("name", f"agent-sandbox-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}")

        try:
            # Prepare sandbox specification
            sandbox_spec = {
                "namespace": namespace,
                "runtime": runtime,
                "name": name,
                "resources": {
                    "cpu_cores": resources.get("cpu_cores", 1.0),
                    "memory_gb": resources.get("memory_gb", 1.0),
                    "disk_gb": resources.get("disk_gb", 2.0),
                    "max_duration_hours": resources.get("max_duration_hours", 1.0),
                    "network_egress": resources.get("network_egress", False),
                },
                "metadata": {
                    "created_by": "agent",
                    "created_at": datetime.now(UTC).isoformat(),
                    "purpose": "agent_fog_computation",
                },
            }

            # Submit to fog gateway
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.fog_gateway_url}/v1/fog/sandboxes",
                    json=sandbox_spec,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 201:
                        result = await response.json()

                        logger.info(f"Created fog sandbox: {result.get('sandbox_id')} in namespace {namespace}")

                        return {
                            "status": "success",
                            "sandbox_id": result.get("sandbox_id"),
                            "endpoint": result.get("endpoint"),
                            "namespace": namespace,
                            "runtime": runtime,
                            "message": f"Sandbox created successfully in namespace {namespace}",
                            "resources": sandbox_spec["resources"],
                        }

                    elif response.status == 403:
                        error_data = await response.json()
                        return {
                            "status": "error",
                            "message": f"Access denied: {error_data.get('message', 'Insufficient permissions')}",
                            "sandbox_id": None,
                            "violations": error_data.get("violations", []),
                        }

                    elif response.status == 429:
                        error_data = await response.json()
                        return {
                            "status": "error",
                            "message": f"Quota exceeded: {error_data.get('message', 'Resource limits reached')}",
                            "sandbox_id": None,
                            "quota_info": error_data.get("quota_status", {}),
                        }

                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "message": f"Fog gateway error ({response.status}): {error_text}",
                            "sandbox_id": None,
                        }

        except aiohttp.ClientError as e:
            logger.error(f"Network error creating sandbox: {e}")
            return {"status": "error", "message": f"Network error: {str(e)}", "sandbox_id": None}

        except Exception as e:
            logger.error(f"Unexpected error creating sandbox: {e}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}", "sandbox_id": None}


class RunJobTool(MCPTool):
    """MCP tool for submitting jobs to fog gateway with namespace validation"""

    def __init__(self, fog_gateway_url: str = "http://localhost:8080"):
        super().__init__("run_job", "Submit job to fog gateway for remote execution")
        self.fog_gateway_url = fog_gateway_url.rstrip("/")

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Submit a job for execution in fog network

        Args:
            parameters:
                namespace: str - Namespace for RBAC and quota enforcement
                image: str - Container image or WASM module reference
                args: list - Command line arguments
                env: dict - Environment variables
                resources: dict - Resource requirements
                input_data: str|bytes - Input data for job
                timeout_s: int - Job timeout in seconds

        Returns:
            dict: {
                "status": "success" | "error",
                "job_id": str,
                "namespace": str,
                "estimated_cost": float,
                "message": str
            }
        """
        self.log_usage()

        # Validate required parameters
        namespace = parameters.get("namespace")
        image = parameters.get("image")

        if not namespace:
            return {"status": "error", "message": "namespace parameter is required", "job_id": None}

        if not image:
            return {"status": "error", "message": "image parameter is required", "job_id": None}

        try:
            # Prepare job specification
            job_spec = {
                "namespace": namespace,
                "runtime": parameters.get("runtime", "wasi"),
                "image": image,
                "args": parameters.get("args", []),
                "env": parameters.get("env", {}),
                "resources": {
                    "cpu_cores": parameters.get("resources", {}).get("cpu_cores", 1.0),
                    "memory_gb": parameters.get("resources", {}).get("memory_gb", 1.0),
                    "disk_gb": parameters.get("resources", {}).get("disk_gb", 2.0),
                    "max_duration_s": parameters.get("timeout_s", 300),
                    "network_egress": parameters.get("resources", {}).get("network_egress", False),
                },
                "input_data": parameters.get("input_data", ""),
                "metadata": {
                    "submitted_by": "agent",
                    "submitted_at": datetime.now(UTC).isoformat(),
                    "source": "agent_fog_tools",
                },
            }

            # Submit job to fog gateway
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.fog_gateway_url}/v1/fog/jobs", json=job_spec, headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 201:
                        result = await response.json()

                        logger.info(f"Submitted fog job: {result.get('job_id')} in namespace {namespace}")

                        return {
                            "status": "success",
                            "job_id": result.get("job_id"),
                            "namespace": namespace,
                            "runtime": job_spec["runtime"],
                            "estimated_cost": result.get("estimated_cost", 0.0),
                            "message": f"Job submitted successfully to namespace {namespace}",
                            "tracking_url": f"{self.fog_gateway_url}/v1/fog/jobs/{result.get('job_id')}/status",
                        }

                    elif response.status == 403:
                        error_data = await response.json()
                        return {
                            "status": "error",
                            "message": f"Job validation failed: {error_data.get('message', 'Access denied')}",
                            "job_id": None,
                            "violations": error_data.get("violations", []),
                        }

                    elif response.status == 429:
                        error_data = await response.json()
                        return {
                            "status": "error",
                            "message": f"Quota exceeded: {error_data.get('message', 'Resource limits reached')}",
                            "job_id": None,
                            "quota_info": error_data.get("quota_status", {}),
                        }

                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "message": f"Fog gateway error ({response.status}): {error_text}",
                            "job_id": None,
                        }

        except aiohttp.ClientError as e:
            logger.error(f"Network error submitting job: {e}")
            return {"status": "error", "message": f"Network error: {str(e)}", "job_id": None}

        except Exception as e:
            logger.error(f"Unexpected error submitting job: {e}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}", "job_id": None}


class StreamLogsTool(MCPTool):
    """MCP tool for real-time log streaming from remote jobs"""

    def __init__(self, fog_gateway_url: str = "http://localhost:8080"):
        super().__init__("stream_logs", "Stream real-time logs from fog job execution")
        self.fog_gateway_url = fog_gateway_url.rstrip("/")

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Stream logs from running fog job

        Args:
            parameters:
                job_id: str - Job ID to stream logs from
                tail_lines: int - Number of recent lines to include
                follow: bool - Whether to follow logs in real-time
                timeout_s: int - Timeout for streaming

        Returns:
            dict: {
                "status": "success" | "error",
                "job_id": str,
                "logs": list - Log lines with timestamps
                "job_status": str,
                "message": str
            }
        """
        self.log_usage()

        # Validate required parameters
        job_id = parameters.get("job_id")
        if not job_id:
            return {"status": "error", "message": "job_id parameter is required", "logs": []}

        tail_lines = parameters.get("tail_lines", 50)
        follow = parameters.get("follow", False)
        timeout_s = parameters.get("timeout_s", 30)

        try:
            logs = []
            job_status = "unknown"

            # Build query parameters
            params = {"tail": tail_lines, "follow": "true" if follow else "false"}

            async with aiohttp.ClientSession() as session:
                # Get job status first
                async with session.get(f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/status") as status_response:
                    if status_response.status == 200:
                        status_data = await status_response.json()
                        job_status = status_data.get("status", "unknown")
                    elif status_response.status == 404:
                        return {"status": "error", "message": f"Job {job_id} not found", "logs": [], "job_id": job_id}

                # Stream logs
                if follow:
                    # Real-time streaming
                    async with session.get(
                        f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/logs", params=params
                    ) as response:
                        if response.status == 200:
                            async for line in response.content:
                                if line:
                                    try:
                                        log_entry = json.loads(line.decode())
                                        logs.append(log_entry)
                                    except json.JSONDecodeError:
                                        # Handle plain text logs
                                        logs.append(
                                            {
                                                "timestamp": datetime.now(UTC).isoformat(),
                                                "level": "INFO",
                                                "message": line.decode().strip(),
                                            }
                                        )

                                # Timeout check for streaming
                                if len(logs) > 1000:  # Prevent memory issues
                                    break
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "message": f"Failed to stream logs ({response.status}): {error_text}",
                                "logs": [],
                                "job_id": job_id,
                            }
                else:
                    # One-time log fetch
                    async with session.get(
                        f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/logs", params=params
                    ) as response:
                        if response.status == 200:
                            log_data = await response.json()
                            logs = log_data.get("logs", [])
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "message": f"Failed to fetch logs ({response.status}): {error_text}",
                                "logs": [],
                                "job_id": job_id,
                            }

            logger.info(f"Retrieved {len(logs)} log entries for job {job_id}")

            return {
                "status": "success",
                "job_id": job_id,
                "logs": logs,
                "job_status": job_status,
                "log_count": len(logs),
                "message": f"Retrieved {len(logs)} log entries for job {job_id}",
            }

        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": f"Timeout after {timeout_s}s streaming logs for job {job_id}",
                "logs": logs,  # Return partial logs
                "job_id": job_id,
            }

        except aiohttp.ClientError as e:
            logger.error(f"Network error streaming logs: {e}")
            return {"status": "error", "message": f"Network error: {str(e)}", "logs": [], "job_id": job_id}

        except Exception as e:
            logger.error(f"Unexpected error streaming logs: {e}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}", "logs": [], "job_id": job_id}


class FetchArtifactsTool(MCPTool):
    """MCP tool for downloading results and outputs from completed fog jobs"""

    def __init__(self, fog_gateway_url: str = "http://localhost:8080"):
        super().__init__("fetch_artifacts", "Download results and outputs from completed fog jobs")
        self.fog_gateway_url = fog_gateway_url.rstrip("/")

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Fetch artifacts from completed fog job

        Args:
            parameters:
                job_id: str - Job ID to fetch artifacts from
                artifact_types: list - Types of artifacts to fetch (stdout, stderr, files, metrics)
                download_files: bool - Whether to download file artifacts

        Returns:
            dict: {
                "status": "success" | "error",
                "job_id": str,
                "artifacts": dict - Artifact data organized by type
                "job_result": dict - Complete job execution result
                "message": str
            }
        """
        self.log_usage()

        # Validate required parameters
        job_id = parameters.get("job_id")
        if not job_id:
            return {"status": "error", "message": "job_id parameter is required", "artifacts": {}}

        artifact_types = parameters.get("artifact_types", ["stdout", "stderr", "metrics"])
        download_files = parameters.get("download_files", False)

        try:
            artifacts = {}
            job_result = {}

            async with aiohttp.ClientSession() as session:
                # Get job result first
                async with session.get(f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/result") as result_response:
                    if result_response.status == 200:
                        job_result = await result_response.json()

                        # Extract basic artifacts from job result
                        if "stdout" in artifact_types and "stdout" in job_result:
                            artifacts["stdout"] = job_result["stdout"]

                        if "stderr" in artifact_types and "stderr" in job_result:
                            artifacts["stderr"] = job_result["stderr"]

                        if "metrics" in artifact_types and "resource_usage" in job_result:
                            artifacts["metrics"] = job_result["resource_usage"]

                    elif result_response.status == 404:
                        return {
                            "status": "error",
                            "message": f"Job {job_id} not found",
                            "artifacts": {},
                            "job_id": job_id,
                        }

                    elif result_response.status == 202:
                        return {
                            "status": "error",
                            "message": f"Job {job_id} is still running - artifacts not ready",
                            "artifacts": {},
                            "job_id": job_id,
                            "job_status": "running",
                        }

                    else:
                        error_text = await result_response.text()
                        return {
                            "status": "error",
                            "message": f"Failed to fetch job result ({result_response.status}): {error_text}",
                            "artifacts": {},
                            "job_id": job_id,
                        }

                # Fetch file artifacts if requested
                if download_files and "files" in artifact_types:
                    async with session.get(
                        f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/artifacts"
                    ) as artifacts_response:
                        if artifacts_response.status == 200:
                            # Handle file downloads (simplified - real implementation would handle binary data)
                            file_data = await artifacts_response.json()
                            artifacts["files"] = file_data.get("files", [])
                        else:
                            logger.warning(
                                f"Failed to fetch file artifacts for job {job_id}: {artifacts_response.status}"
                            )
                            artifacts["files"] = []

            # Add job metadata
            artifacts["job_metadata"] = {
                "job_id": job_id,
                "status": job_result.get("status", "unknown"),
                "exit_code": job_result.get("exit_code"),
                "duration_ms": job_result.get("duration_ms", 0),
                "fetched_at": datetime.now(UTC).isoformat(),
            }

            logger.info(f"Fetched {len(artifacts)} artifact types for job {job_id}")

            return {
                "status": "success",
                "job_id": job_id,
                "artifacts": artifacts,
                "job_result": job_result,
                "artifact_count": len(artifacts),
                "message": f"Fetched {len(artifacts)} artifact types for job {job_id}",
            }

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching artifacts: {e}")
            return {"status": "error", "message": f"Network error: {str(e)}", "artifacts": {}, "job_id": job_id}

        except Exception as e:
            logger.error(f"Unexpected error fetching artifacts: {e}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}", "artifacts": {}, "job_id": job_id}


class FogJobStatusTool(MCPTool):
    """MCP tool for checking fog job status and progress"""

    def __init__(self, fog_gateway_url: str = "http://localhost:8080"):
        super().__init__("fog_job_status", "Check status and progress of fog job execution")
        self.fog_gateway_url = fog_gateway_url.rstrip("/")

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Check fog job status

        Args:
            parameters:
                job_id: str - Job ID to check status for
                include_logs: bool - Include recent log entries

        Returns:
            dict: {
                "status": "success" | "error",
                "job_id": str,
                "job_status": str,
                "progress": dict,
                "resource_usage": dict,
                "recent_logs": list,
                "message": str
            }
        """
        self.log_usage()

        # Validate required parameters
        job_id = parameters.get("job_id")
        if not job_id:
            return {"status": "error", "message": "job_id parameter is required", "job_status": "unknown"}

        include_logs = parameters.get("include_logs", False)

        try:
            async with aiohttp.ClientSession() as session:
                # Get job status
                async with session.get(f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/status") as response:
                    if response.status == 200:
                        status_data = await response.json()

                        result = {
                            "status": "success",
                            "job_id": job_id,
                            "job_status": status_data.get("status", "unknown"),
                            "progress": status_data.get("progress", {}),
                            "resource_usage": status_data.get("resource_usage", {}),
                            "message": f"Job {job_id} status: {status_data.get('status', 'unknown')}",
                        }

                        # Include recent logs if requested
                        if include_logs:
                            log_tool = StreamLogsTool(self.fog_gateway_url)
                            log_result = await log_tool.execute({"job_id": job_id, "tail_lines": 10, "follow": False})

                            if log_result["status"] == "success":
                                result["recent_logs"] = log_result["logs"]
                            else:
                                result["recent_logs"] = []
                                result["log_error"] = log_result["message"]

                        return result

                    elif response.status == 404:
                        return {
                            "status": "error",
                            "message": f"Job {job_id} not found",
                            "job_id": job_id,
                            "job_status": "not_found",
                        }

                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "message": f"Failed to get job status ({response.status}): {error_text}",
                            "job_id": job_id,
                            "job_status": "unknown",
                        }

        except aiohttp.ClientError as e:
            logger.error(f"Network error checking job status: {e}")
            return {"status": "error", "message": f"Network error: {str(e)}", "job_id": job_id, "job_status": "unknown"}

        except Exception as e:
            logger.error(f"Unexpected error checking job status: {e}")
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "job_id": job_id,
                "job_status": "unknown",
            }


# Export all MCP fog tools for agent integration
__all__ = ["CreateSandboxTool", "RunJobTool", "StreamLogsTool", "FetchArtifactsTool", "FogJobStatusTool"]
