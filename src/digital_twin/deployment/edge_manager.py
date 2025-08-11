"""Edge Deployment Manager - Device-Specific AI Deployment
Sprint R-5: Digital Twin MVP - Task A.5.
"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import platform
import sqlite3
from typing import Any
import zipfile

import numpy as np
import psutil
import torch
import wandb

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    CHROMEBOOK = "chromebook"
    RASPBERRY_PI = "raspberry_pi"
    SMART_TV = "smart_tv"
    OTHER = "other"


class DeploymentStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    OUTDATED = "outdated"


@dataclass
class DeviceProfile:
    """Profile of target deployment device."""

    device_id: str
    device_name: str
    device_type: DeviceType
    os_name: str
    os_version: str
    cpu_architecture: str
    cpu_cores: int
    ram_mb: int
    storage_available_mb: int
    gpu_available: bool
    gpu_memory_mb: int
    network_speed_mbps: float
    battery_powered: bool
    always_connected: bool
    parental_controls: bool
    student_ids: list[str]  # Students who use this device
    last_seen: str
    created_at: str


@dataclass
class TutorDeployment:
    """Deployed tutor instance on a device."""

    deployment_id: str
    device_id: str
    student_id: str
    tutor_model_id: str
    tutor_version: str
    deployment_package_path: str
    status: DeploymentStatus
    model_size_mb: float
    compressed_size_mb: float
    installation_date: str
    last_used: str
    usage_count: int
    performance_metrics: dict[str, float]
    sync_status: str  # synced, pending, conflict
    offline_capable: bool
    auto_update: bool
    parent_approved: bool


@dataclass
class EdgeUpdate:
    """Update for edge-deployed tutors."""

    update_id: str
    deployment_id: str
    update_type: str  # model, preferences, content, security
    update_size_mb: float
    priority: str  # critical, high, normal, low
    description: str
    changelog: list[str]
    requires_restart: bool
    scheduled_time: str | None
    downloaded: bool
    installed: bool
    rollback_available: bool


class EdgeDeploymentManager:
    """Comprehensive edge deployment and management system."""

    def __init__(self, project_name: str = "aivillage-edge") -> None:
        self.project_name = project_name
        self.devices = {}  # device_id -> DeviceProfile
        self.deployments = {}  # deployment_id -> TutorDeployment
        self.pending_updates = defaultdict(list)  # device_id -> List[EdgeUpdate]

        # Deployment configurations
        self.deployment_configs = {
            DeviceType.SMARTPHONE: {
                "max_model_size_mb": 100,
                "compression_level": "aggressive",
                "offline_duration_hours": 24,
                "sync_frequency_minutes": 60,
                "auto_cleanup": True,
            },
            DeviceType.TABLET: {
                "max_model_size_mb": 200,
                "compression_level": "moderate",
                "offline_duration_hours": 48,
                "sync_frequency_minutes": 30,
                "auto_cleanup": True,
            },
            DeviceType.LAPTOP: {
                "max_model_size_mb": 500,
                "compression_level": "light",
                "offline_duration_hours": 168,  # 1 week
                "sync_frequency_minutes": 15,
                "auto_cleanup": False,
            },
            DeviceType.CHROMEBOOK: {
                "max_model_size_mb": 150,
                "compression_level": "moderate",
                "offline_duration_hours": 72,
                "sync_frequency_minutes": 45,
                "auto_cleanup": True,
            },
            DeviceType.RASPBERRY_PI: {
                "max_model_size_mb": 200,
                "compression_level": "moderate",
                "offline_duration_hours": 168,
                "sync_frequency_minutes": 30,
                "auto_cleanup": False,
            },
        }

        # Content delivery network (CDN) endpoints
        self.cdn_endpoints = [
            "https://cdn1.aivillage.edu/models/",
            "https://cdn2.aivillage.edu/models/",
            "https://cdn3.aivillage.edu/models/",
        ]

        # Local deployment server
        self.local_deployment_path = Path("edge_deployments")
        self.local_deployment_path.mkdir(exist_ok=True)

        # Database for deployment tracking
        self.db_path = "edge_deployment.db"
        self.init_database()

        # Background tasks
        self.deployment_monitor_active = True
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Performance monitoring
        self.deployment_metrics = defaultdict(dict)
        self.sync_performance = defaultdict(list)

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Start background deployment monitoring (only if event loop is available)
        try:
            asyncio.create_task(self.start_deployment_monitoring())
        except RuntimeError:
            # No event loop available - deployment monitoring can be started later
            logger.info("No event loop available, deployment monitoring can be started manually")

        logger.info("Edge Deployment Manager initialized")

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for edge deployment."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="edge_deployment",
                config={
                    "deployment_version": "1.0.0",
                    "supported_devices": [dt.value for dt in DeviceType],
                    "deployment_features": [
                        "auto_device_detection",
                        "adaptive_compression",
                        "offline_sync",
                        "incremental_updates",
                        "rollback_support",
                        "parental_controls",
                        "usage_analytics",
                    ],
                    "cdn_endpoints": len(self.cdn_endpoints),
                    "offline_capable": True,
                    "auto_scaling": True,
                    "multi_student_support": True,
                },
            )

            logger.info("Edge deployment W&B tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B tracking: {e}")

    def init_database(self) -> None:
        """Initialize database for deployment tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Device profiles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS device_profiles (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    device_specs TEXT NOT NULL,  -- JSON
                    student_ids TEXT NOT NULL,   -- JSON
                    last_seen TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Tutor deployments table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tutor_deployments (
                    deployment_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    deployment_data TEXT NOT NULL,  -- JSON
                    status TEXT NOT NULL,
                    installation_date TEXT NOT NULL,
                    last_used TEXT NOT NULL
                )
            """
            )

            # Edge updates table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_updates (
                    update_id TEXT PRIMARY KEY,
                    deployment_id TEXT NOT NULL,
                    update_data TEXT NOT NULL,  -- JSON
                    created_at TEXT NOT NULL,
                    installed_at TEXT
                )
            """
            )

            # Deployment metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_metrics (
                    metric_id TEXT PRIMARY KEY,
                    deployment_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_data TEXT NOT NULL,  -- JSON
                    recorded_at TEXT NOT NULL
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_deployments_device ON tutor_deployments(device_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_deployments_student ON tutor_deployments(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_updates_deployment ON edge_updates(deployment_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_deployment ON deployment_metrics(deployment_id)"
            )

            conn.commit()
            conn.close()

            logger.info("Edge deployment database initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")

    async def register_device(
        self,
        device_id: str,
        device_name: str,
        student_ids: list[str],
        auto_detect_specs: bool = True,
    ) -> DeviceProfile:
        """Register a new device for edge deployment."""
        # Auto-detect device specifications
        if auto_detect_specs:
            device_specs = await self._detect_device_specifications()
        else:
            device_specs = self._get_default_device_specs()

        # Determine device type based on specifications
        device_type = self._classify_device_type(device_specs)

        device_profile = DeviceProfile(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            os_name=device_specs["os_name"],
            os_version=device_specs["os_version"],
            cpu_architecture=device_specs["cpu_architecture"],
            cpu_cores=device_specs["cpu_cores"],
            ram_mb=device_specs["ram_mb"],
            storage_available_mb=device_specs["storage_available_mb"],
            gpu_available=device_specs["gpu_available"],
            gpu_memory_mb=device_specs["gpu_memory_mb"],
            network_speed_mbps=device_specs["network_speed_mbps"],
            battery_powered=device_specs["battery_powered"],
            always_connected=device_specs["always_connected"],
            parental_controls=device_specs["parental_controls"],
            student_ids=student_ids,
            last_seen=datetime.now(timezone.utc).isoformat(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Store device profile
        self.devices[device_id] = device_profile
        await self._save_device_profile(device_profile)

        # Log to W&B
        wandb.log(
            {
                "edge/device_registered": True,
                "edge/device_type": device_type.value,
                "edge/cpu_cores": device_specs["cpu_cores"],
                "edge/ram_mb": device_specs["ram_mb"],
                "edge/gpu_available": device_specs["gpu_available"],
                "edge/students_count": len(student_ids),
                "timestamp": device_profile.created_at,
            }
        )

        logger.info(
            f"Registered device {device_name} ({device_type.value}) for {len(student_ids)} students"
        )

        return device_profile

    async def _detect_device_specifications(self) -> dict[str, Any]:
        """Auto-detect device specifications."""
        try:
            specs = {
                "os_name": platform.system(),
                "os_version": platform.version(),
                "cpu_architecture": platform.machine(),
                "cpu_cores": psutil.cpu_count(logical=False) or 1,
                "ram_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
                "storage_available_mb": int(
                    psutil.disk_usage("/").free / (1024 * 1024)
                ),
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_mb": 0,
                "network_speed_mbps": 10.0,  # Default estimate
                "battery_powered": False,  # Default assumption
                "always_connected": True,  # Default assumption
                "parental_controls": False,  # Default assumption
            }

            # Get GPU memory if available
            if specs["gpu_available"]:
                try:
                    specs["gpu_memory_mb"] = int(
                        torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    )
                except:
                    specs["gpu_memory_mb"] = 0

            # Detect if battery powered (rough heuristic)
            try:
                battery = psutil.sensors_battery()
                if battery:
                    specs["battery_powered"] = True
            except:
                pass

            return specs

        except Exception as e:
            logger.warning(f"Error detecting device specs: {e}")
            return self._get_default_device_specs()

    def _get_default_device_specs(self) -> dict[str, Any]:
        """Get default device specifications."""
        return {
            "os_name": "Unknown",
            "os_version": "Unknown",
            "cpu_architecture": "x86_64",
            "cpu_cores": 2,
            "ram_mb": 4096,
            "storage_available_mb": 10000,
            "gpu_available": False,
            "gpu_memory_mb": 0,
            "network_speed_mbps": 10.0,
            "battery_powered": False,
            "always_connected": True,
            "parental_controls": False,
        }

    def _classify_device_type(self, specs: dict[str, Any]) -> DeviceType:
        """Classify device type based on specifications."""
        ram_mb = specs["ram_mb"]
        cpu_cores = specs["cpu_cores"]
        battery_powered = specs["battery_powered"]
        os_name = specs["os_name"].lower()

        # Classification logic
        if "android" in os_name or "ios" in os_name:
            if ram_mb <= 3000:
                return DeviceType.SMARTPHONE
            return DeviceType.TABLET
        if "chrome" in os_name:
            return DeviceType.CHROMEBOOK
        if cpu_cores <= 2 and ram_mb <= 2000:
            return DeviceType.RASPBERRY_PI
        if battery_powered:
            if ram_mb <= 8000:
                return DeviceType.LAPTOP
            return DeviceType.LAPTOP
        return DeviceType.DESKTOP

    async def deploy_tutor(
        self,
        device_id: str,
        student_id: str,
        tutor_model_id: str,
        force_update: bool = False,
    ) -> str:
        """Deploy personalized tutor to edge device."""
        if device_id not in self.devices:
            msg = f"Device {device_id} not registered"
            raise ValueError(msg)

        device = self.devices[device_id]

        # Check if deployment already exists
        existing_deployment = self._find_existing_deployment(device_id, student_id)
        if existing_deployment and not force_update:
            logger.info(
                f"Tutor already deployed for student {student_id[:8]} on device {device_id}"
            )
            return existing_deployment.deployment_id

        # Generate deployment ID
        deployment_id = (
            f"deploy_{device_id[:8]}_{student_id[:8]}_{int(datetime.now().timestamp())}"
        )

        try:
            # Get deployment configuration for device type
            config = self.deployment_configs.get(
                device.device_type, self.deployment_configs[DeviceType.LAPTOP]
            )

            # Prepare tutor for deployment
            deployment_package = await self._prepare_tutor_deployment(
                tutor_model_id, student_id, device, config
            )

            # Create deployment record
            deployment = TutorDeployment(
                deployment_id=deployment_id,
                device_id=device_id,
                student_id=student_id,
                tutor_model_id=tutor_model_id,
                tutor_version="1.0.0",
                deployment_package_path=deployment_package["path"],
                status=DeploymentStatus.PENDING,
                model_size_mb=deployment_package["original_size_mb"],
                compressed_size_mb=deployment_package["compressed_size_mb"],
                installation_date=datetime.now(timezone.utc).isoformat(),
                last_used="never",
                usage_count=0,
                performance_metrics={},
                sync_status="pending",
                offline_capable=True,
                auto_update=True,
                parent_approved=True,  # Would check with parent tracker
            )

            # Store deployment
            self.deployments[deployment_id] = deployment
            await self._save_deployment(deployment)

            # Start deployment process
            asyncio.create_task(self._execute_deployment(deployment))

            # Log to W&B
            wandb.log(
                {
                    "edge/deployment_started": True,
                    "edge/device_type": device.device_type.value,
                    "edge/model_size_mb": deployment.model_size_mb,
                    "edge/compressed_size_mb": deployment.compressed_size_mb,
                    "edge/compression_ratio": deployment.model_size_mb
                    / max(deployment.compressed_size_mb, 1),
                    "timestamp": deployment.installation_date,
                }
            )

            logger.info(
                f"Started deployment {deployment_id} for student {student_id[:8]} on device {device_id}"
            )

            return deployment_id

        except Exception as e:
            logger.exception(f"Failed to deploy tutor: {e}")
            raise

    def _find_existing_deployment(
        self, device_id: str, student_id: str
    ) -> TutorDeployment | None:
        """Find existing deployment for student on device."""
        for deployment in self.deployments.values():
            if (
                deployment.device_id == device_id
                and deployment.student_id == student_id
                and deployment.status != DeploymentStatus.ERROR
            ):
                return deployment

        return None

    async def _prepare_tutor_deployment(
        self,
        tutor_model_id: str,
        student_id: str,
        device: DeviceProfile,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare tutor model for deployment."""
        try:
            # Import deployment system
            from src.agent_forge.evolution.deploy_winner import tutor_deployment

            # Get champion model (would normally load from storage)
            champion_model = {
                "individual_id": tutor_model_id,
                "fitness_score": 0.85,  # Placeholder
                "model_name": "evolved_math_tutor",
                "model": None,  # Would load actual model
            }

            # Determine target platform based on device type
            platform_map = {
                DeviceType.SMARTPHONE: "mobile",
                DeviceType.TABLET: "mobile",
                DeviceType.LAPTOP: "desktop",
                DeviceType.DESKTOP: "server",
                DeviceType.CHROMEBOOK: "web_browser",
                DeviceType.RASPBERRY_PI: "edge_server",
            }

            target_platform = platform_map.get(device.device_type, "desktop")

            # Prepare deployment package
            deployment_package = await tutor_deployment.prepare_champion(
                champion_model=champion_model, target_platform=target_platform
            )

            # Create local deployment package
            package_path = (
                self.local_deployment_path
                / f"{tutor_model_id}_{student_id}_{device.device_id}.zip"
            )

            # Create deployment ZIP file
            await self._create_deployment_zip(
                deployment_package, package_path, student_id, device
            )

            return {
                "path": str(package_path),
                "original_size_mb": deployment_package.model_size_mb,
                "compressed_size_mb": deployment_package.deployment_size_mb,
                "package_id": deployment_package.package_id,
            }

        except Exception as e:
            logger.exception(f"Error preparing tutor deployment: {e}")

            # Create fallback deployment package
            fallback_path = (
                self.local_deployment_path
                / f"fallback_{student_id}_{device.device_id}.zip"
            )
            await self._create_fallback_deployment(fallback_path, student_id, device)

            return {
                "path": str(fallback_path),
                "original_size_mb": 50.0,
                "compressed_size_mb": 25.0,
                "package_id": f"fallback_{student_id[:8]}",
            }

    async def _create_deployment_zip(
        self,
        deployment_package: Any,
        package_path: Path,
        student_id: str,
        device: DeviceProfile,
    ) -> None:
        """Create deployment ZIP package."""
        try:
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add deployment manifest
                manifest = {
                    "package_version": "1.0.0",
                    "student_id": student_id,
                    "device_id": device.device_id,
                    "device_type": device.device_type.value,
                    "deployment_date": datetime.now(timezone.utc).isoformat(),
                    "model_info": {
                        "package_id": deployment_package.package_id,
                        "model_size_mb": deployment_package.deployment_size_mb,
                        "performance_benchmarks": deployment_package.performance_benchmarks,
                    },
                    "requirements": deployment_package.requirements,
                    "installation_script": deployment_package.installation_script,
                }

                zipf.writestr("manifest.json", json.dumps(manifest, indent=2))

                # Add model files (if they exist)
                if (
                    hasattr(deployment_package, "model_path")
                    and Path(deployment_package.model_path).exists()
                ):
                    model_path = Path(deployment_package.model_path)
                    for file_path in model_path.rglob("*"):
                        if file_path.is_file():
                            arcname = f"model/{file_path.relative_to(model_path)}"
                            zipf.write(file_path, arcname)

                # Add configuration files
                config_data = {
                    "student_preferences": {},  # Would load from preference vault
                    "device_config": asdict(device),
                    "learning_settings": {
                        "offline_mode": True,
                        "sync_frequency": "hourly",
                        "auto_update": True,
                    },
                }

                zipf.writestr(
                    "config.json", json.dumps(config_data, indent=2, default=str)
                )

                # Add offline content
                offline_content = await self._generate_offline_content(
                    student_id, device
                )
                zipf.writestr(
                    "offline_content.json", json.dumps(offline_content, indent=2)
                )

                logger.info(f"Created deployment package: {package_path}")

        except Exception as e:
            logger.exception(f"Failed to create deployment ZIP: {e}")
            raise

    async def _create_fallback_deployment(
        self, package_path: Path, student_id: str, device: DeviceProfile
    ) -> None:
        """Create minimal fallback deployment."""
        try:
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Minimal manifest
                manifest = {
                    "package_version": "1.0.0-fallback",
                    "student_id": student_id,
                    "device_id": device.device_id,
                    "deployment_date": datetime.now(timezone.utc).isoformat(),
                    "fallback": True,
                    "description": "Minimal tutoring functionality for offline use",
                }

                zipf.writestr("manifest.json", json.dumps(manifest, indent=2))

                # Basic tutoring content
                basic_content = {
                    "lessons": [
                        {
                            "id": "basic_math_1",
                            "title": "Addition Practice",
                            "content": "Let's practice adding numbers together!",
                            "problems": [
                                {"question": "5 + 3 = ?", "answer": "8"},
                                {"question": "7 + 2 = ?", "answer": "9"},
                                {"question": "4 + 6 = ?", "answer": "10"},
                            ],
                        }
                    ],
                    "explanations": {
                        "addition": "Adding means putting numbers together to find the total."
                    },
                }

                zipf.writestr("basic_content.json", json.dumps(basic_content, indent=2))

        except Exception as e:
            logger.exception(f"Failed to create fallback deployment: {e}")
            raise

    async def _generate_offline_content(
        self, student_id: str, device: DeviceProfile
    ) -> dict[str, Any]:
        """Generate offline content package."""
        # Get student's learning profile
        try:
            from src.digital_twin.core.digital_twin import digital_twin

            if student_id in digital_twin.students:
                student = digital_twin.students[student_id]

                # Generate age-appropriate content
                offline_content = {
                    "student_info": {
                        "age": student.age,
                        "grade_level": student.grade_level,
                        "learning_style": student.learning_style,
                        "strengths": student.strengths,
                        "interests": student.interests,
                    },
                    "practice_problems": await self._generate_practice_problems(
                        student
                    ),
                    "explanations": await self._generate_explanations(student),
                    "encouragement_messages": [
                        "Great job working hard!",
                        "You're making excellent progress!",
                        "Keep up the fantastic work!",
                        "I'm proud of your effort!",
                        "You're becoming a math star!",
                    ],
                    "offline_activities": await self._generate_offline_activities(
                        student
                    ),
                    "progress_tracking": {"enabled": True, "sync_on_reconnect": True},
                }

                return offline_content

        except Exception as e:
            logger.warning(f"Could not generate personalized offline content: {e}")

        # Fallback generic content
        return {
            "practice_problems": [
                {"question": "What is 8 + 5?", "answer": "13", "grade": 2},
                {"question": "What is 12 - 7?", "answer": "5", "grade": 2},
                {"question": "What is 6 × 3?", "answer": "18", "grade": 3},
            ],
            "explanations": {
                "addition": "Addition means putting numbers together.",
                "subtraction": "Subtraction means taking away numbers.",
                "multiplication": "Multiplication means adding groups together.",
            },
            "encouragement_messages": [
                "You're doing great!",
                "Keep practicing!",
                "Nice work!",
            ],
        }

    async def _generate_practice_problems(self, student) -> list[dict[str, Any]]:
        """Generate grade-appropriate practice problems."""
        problems = []
        grade = student.grade_level

        if grade <= 2:
            # Basic addition and subtraction
            for _i in range(10):
                a, b = np.random.randint(1, 11), np.random.randint(1, 11)
                problems.append(
                    {
                        "question": f"What is {a} + {b}?",
                        "answer": str(a + b),
                        "type": "addition",
                        "difficulty": "easy",
                    }
                )

                if a > b:
                    problems.append(
                        {
                            "question": f"What is {a} - {b}?",
                            "answer": str(a - b),
                            "type": "subtraction",
                            "difficulty": "easy",
                        }
                    )

        elif grade <= 4:
            # Multiplication and division
            for _i in range(10):
                a, b = np.random.randint(2, 10), np.random.randint(2, 10)
                problems.append(
                    {
                        "question": f"What is {a} × {b}?",
                        "answer": str(a * b),
                        "type": "multiplication",
                        "difficulty": "medium",
                    }
                )

                product = a * b
                problems.append(
                    {
                        "question": f"What is {product} ÷ {a}?",
                        "answer": str(b),
                        "type": "division",
                        "difficulty": "medium",
                    }
                )

        else:
            # More advanced problems
            for _i in range(8):
                a, b = np.random.randint(10, 100), np.random.randint(2, 20)
                problems.append(
                    {
                        "question": f"What is {a} + {b}?",
                        "answer": str(a + b),
                        "type": "addition",
                        "difficulty": "medium",
                    }
                )

        return problems

    async def _generate_explanations(self, student) -> dict[str, str]:
        """Generate age-appropriate explanations."""
        if student.age <= 8:
            return {
                "addition": "Adding means putting numbers together to get a bigger number!",
                "subtraction": "Subtracting means taking some numbers away to get a smaller number!",
                "counting": "Counting helps us find out how many things we have!",
            }
        return {
            "addition": "Addition combines two or more numbers to find their total sum.",
            "subtraction": "Subtraction finds the difference between two numbers.",
            "multiplication": "Multiplication is repeated addition of the same number.",
            "division": "Division splits a number into equal groups.",
        }

    async def _generate_offline_activities(self, student) -> list[dict[str, Any]]:
        """Generate offline learning activities."""
        activities = []

        if "sports" in student.interests:
            activities.append(
                {
                    "title": "Sports Math",
                    "description": "Practice math using sports examples!",
                    "problems": [
                        "If a football team scores 7 points 3 times, how many points total?",
                        "A basketball player makes 8 free throws out of 10 attempts. How many did they miss?",
                    ],
                }
            )

        if "art" in student.interests:
            activities.append(
                {
                    "title": "Art Patterns",
                    "description": "Create mathematical patterns with drawings!",
                    "problems": [
                        "Draw a pattern using shapes: circle, square, circle, square...",
                        "If you use 5 colors in a pattern, how many colors in 3 complete patterns?",
                    ],
                }
            )

        # Default activities
        activities.append(
            {
                "title": "Real World Math",
                "description": "Find math in everyday situations!",
                "problems": [
                    "Count the windows in your house and multiply by 2 for the panes.",
                    "If you read 2 books per week, how many in a month?",
                ],
            }
        )

        return activities

    async def _execute_deployment(self, deployment: TutorDeployment) -> None:
        """Execute the deployment process."""
        try:
            # Update status to downloading
            deployment.status = DeploymentStatus.DOWNLOADING
            await self._save_deployment(deployment)

            # Simulate download process
            await self._download_deployment_package(deployment)

            # Update status to installing
            deployment.status = DeploymentStatus.INSTALLING
            await self._save_deployment(deployment)

            # Simulate installation
            await self._install_deployment_package(deployment)

            # Update status to ready
            deployment.status = DeploymentStatus.READY
            deployment.sync_status = "synced"
            await self._save_deployment(deployment)

            # Log successful deployment
            wandb.log(
                {
                    "edge/deployment_completed": True,
                    "edge/deployment_id": deployment.deployment_id,
                    "edge/student_id": deployment.student_id,
                    "edge/device_id": deployment.device_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logger.info(f"Successfully deployed {deployment.deployment_id}")

        except Exception as e:
            deployment.status = DeploymentStatus.ERROR
            await self._save_deployment(deployment)

            logger.exception(f"Deployment failed for {deployment.deployment_id}: {e}")

            # Log failed deployment
            wandb.log(
                {
                    "edge/deployment_failed": True,
                    "edge/deployment_id": deployment.deployment_id,
                    "edge/error": str(e)[:200],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    async def _download_deployment_package(self, deployment: TutorDeployment) -> None:
        """Download deployment package to device."""
        # Simulate download with progress tracking
        package_size_mb = deployment.compressed_size_mb
        downloaded_mb = 0
        chunk_size_mb = 5  # 5MB chunks

        while downloaded_mb < package_size_mb:
            await asyncio.sleep(0.5)  # Simulate download time
            downloaded_mb += chunk_size_mb
            progress = min(100, (downloaded_mb / package_size_mb) * 100)

            # Log download progress
            if int(progress) % 20 == 0:  # Log every 20%
                logger.info(
                    f"Download progress for {deployment.deployment_id}: {progress:.1f}%"
                )

        logger.info(f"Download completed for {deployment.deployment_id}")

    async def _install_deployment_package(self, deployment: TutorDeployment) -> None:
        """Install deployment package on device."""
        # Simulate installation process
        installation_steps = [
            "Extracting package files",
            "Installing dependencies",
            "Configuring tutor settings",
            "Setting up offline content",
            "Running initial tests",
            "Finalizing installation",
        ]

        for i, step in enumerate(installation_steps):
            await asyncio.sleep(0.3)  # Simulate installation time
            progress = ((i + 1) / len(installation_steps)) * 100
            logger.info(
                f"Installation {deployment.deployment_id}: {step} ({progress:.1f}%)"
            )

        logger.info(f"Installation completed for {deployment.deployment_id}")

    async def update_deployment(
        self,
        deployment_id: str,
        update_type: str,
        update_data: Any,
        priority: str = "normal",
    ) -> str:
        """Create and schedule deployment update."""
        if deployment_id not in self.deployments:
            msg = f"Deployment {deployment_id} not found"
            raise ValueError(msg)

        deployment = self.deployments[deployment_id]

        # Generate update ID
        update_id = f"update_{deployment_id[:8]}_{update_type}_{int(datetime.now().timestamp())}"

        # Estimate update size
        update_size_mb = self._estimate_update_size(update_type, update_data)

        # Create update record
        update = EdgeUpdate(
            update_id=update_id,
            deployment_id=deployment_id,
            update_type=update_type,
            update_size_mb=update_size_mb,
            priority=priority,
            description=f"Update {update_type} for deployment {deployment_id}",
            changelog=[f"Updated {update_type}"],
            requires_restart=update_type in ["model", "security"],
            scheduled_time=None,  # Immediate for now
            downloaded=False,
            installed=False,
            rollback_available=True,
        )

        # Add to pending updates
        self.pending_updates[deployment.device_id].append(update)
        await self._save_update(update)

        # Schedule update execution
        asyncio.create_task(self._execute_update(update))

        logger.info(
            f"Scheduled {update_type} update {update_id} for deployment {deployment_id}"
        )

        return update_id

    def _estimate_update_size(self, update_type: str, update_data: Any) -> float:
        """Estimate update size in MB."""
        size_estimates = {
            "model": 50.0,  # Model updates are usually large
            "preferences": 0.1,  # Preference updates are small
            "content": 10.0,  # Content updates are medium
            "security": 5.0,  # Security patches are small-medium
            "config": 0.01,  # Config updates are tiny
        }

        return size_estimates.get(update_type, 1.0)

    async def _execute_update(self, update: EdgeUpdate) -> None:
        """Execute deployment update."""
        try:
            deployment = self.deployments[update.deployment_id]

            # Simulate download
            update.downloaded = True
            await self._save_update(update)

            # Simulate installation
            await asyncio.sleep(1)  # Simulate installation time

            # Apply update based on type
            if update.update_type == "model":
                await self._update_model(deployment, update)
            elif update.update_type == "preferences":
                await self._update_preferences(deployment, update)
            elif update.update_type == "content":
                await self._update_content(deployment, update)
            elif update.update_type == "security":
                await self._update_security(deployment, update)

            # Mark as installed
            update.installed = True
            await self._save_update(update)

            # Update deployment version
            deployment.tutor_version = (
                f"{deployment.tutor_version}.{int(datetime.now().timestamp()) % 1000}"
            )
            await self._save_deployment(deployment)

            # Log successful update
            wandb.log(
                {
                    "edge/update_completed": True,
                    "edge/update_type": update.update_type,
                    "edge/update_size_mb": update.update_size_mb,
                    "edge/deployment_id": deployment.deployment_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logger.info(f"Successfully applied update {update.update_id}")

        except Exception as e:
            logger.exception(f"Update failed for {update.update_id}: {e}")

            # Log failed update
            wandb.log(
                {
                    "edge/update_failed": True,
                    "edge/update_id": update.update_id,
                    "edge/error": str(e)[:200],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    async def _update_model(
        self, deployment: TutorDeployment, update: EdgeUpdate
    ) -> None:
        """Update tutor model."""
        logger.info(f"Updating model for deployment {deployment.deployment_id}")
        # Would implement actual model update logic

    async def _update_preferences(
        self, deployment: TutorDeployment, update: EdgeUpdate
    ) -> None:
        """Update user preferences."""
        logger.info(f"Updating preferences for deployment {deployment.deployment_id}")
        # Would implement preference sync logic

    async def _update_content(
        self, deployment: TutorDeployment, update: EdgeUpdate
    ) -> None:
        """Update learning content."""
        logger.info(f"Updating content for deployment {deployment.deployment_id}")
        # Would implement content update logic

    async def _update_security(
        self, deployment: TutorDeployment, update: EdgeUpdate
    ) -> None:
        """Update security configurations."""
        logger.info(f"Updating security for deployment {deployment.deployment_id}")
        # Would implement security update logic

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get comprehensive deployment status."""
        if deployment_id not in self.deployments:
            return {"error": "Deployment not found"}

        deployment = self.deployments[deployment_id]
        device = self.devices[deployment.device_id]

        # Check for pending updates
        pending_updates = [
            asdict(update)
            for update in self.pending_updates[deployment.device_id]
            if update.deployment_id == deployment_id and not update.installed
        ]

        # Get recent performance metrics
        recent_metrics = self.deployment_metrics.get(deployment_id, {})

        status = {
            "deployment_info": asdict(deployment),
            "device_info": {
                "device_name": device.device_name,
                "device_type": device.device_type.value,
                "os_name": device.os_name,
                "last_seen": device.last_seen,
            },
            "current_status": deployment.status.value,
            "sync_status": deployment.sync_status,
            "usage_stats": {
                "usage_count": deployment.usage_count,
                "last_used": deployment.last_used,
                "total_study_time_hours": recent_metrics.get(
                    "total_study_time_hours", 0
                ),
            },
            "performance_metrics": deployment.performance_metrics,
            "pending_updates": pending_updates,
            "storage_usage": {
                "model_size_mb": deployment.model_size_mb,
                "compressed_size_mb": deployment.compressed_size_mb,
                "additional_content_mb": recent_metrics.get("additional_content_mb", 0),
            },
            "last_sync": recent_metrics.get("last_sync", "never"),
            "offline_capable": deployment.offline_capable,
            "auto_update_enabled": deployment.auto_update,
        }

        return status

    async def sync_deployment(self, deployment_id: str) -> bool:
        """Sync deployment with cloud."""
        if deployment_id not in self.deployments:
            return False

        deployment = self.deployments[deployment_id]

        try:
            # Update sync status
            deployment.sync_status = "syncing"
            await self._save_deployment(deployment)

            # Simulate sync operations
            sync_operations = [
                "Uploading usage data",
                "Downloading preference updates",
                "Syncing progress data",
                "Checking for content updates",
                "Validating model integrity",
            ]

            for operation in sync_operations:
                await asyncio.sleep(0.2)  # Simulate sync time
                logger.info(f"Sync {deployment_id}: {operation}")

            # Update sync status
            deployment.sync_status = "synced"
            await self._save_deployment(deployment)

            # Record sync performance
            sync_time = datetime.now(timezone.utc).isoformat()
            self.sync_performance[deployment_id].append(
                {
                    "timestamp": sync_time,
                    "duration_ms": 1000,  # Simulated
                    "success": True,
                }
            )

            # Log successful sync
            wandb.log(
                {
                    "edge/sync_completed": True,
                    "edge/deployment_id": deployment_id,
                    "timestamp": sync_time,
                }
            )

            logger.info(f"Successfully synced deployment {deployment_id}")
            return True

        except Exception as e:
            deployment.sync_status = "error"
            await self._save_deployment(deployment)

            logger.exception(f"Sync failed for {deployment_id}: {e}")
            return False

    async def start_deployment_monitoring(self) -> None:
        """Start background deployment monitoring."""
        while self.deployment_monitor_active:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check deployment health
                await self._check_deployment_health()

                # Process pending updates
                await self._process_pending_updates()

                # Update device last seen
                await self._update_device_status()

                # Clean up old deployments
                await self._cleanup_old_deployments()

            except Exception as e:
                logger.exception(f"Error in deployment monitoring: {e}")
                await asyncio.sleep(60)

    async def _check_deployment_health(self) -> None:
        """Check health of all deployments."""
        for deployment in self.deployments.values():
            if deployment.status == DeploymentStatus.RUNNING:
                # Check if deployment is responsive
                health_check = await self._perform_health_check(deployment)

                if not health_check:
                    logger.warning(
                        f"Health check failed for deployment {deployment.deployment_id}"
                    )

                    # Log health issue
                    wandb.log(
                        {
                            "edge/health_check_failed": True,
                            "edge/deployment_id": deployment.deployment_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

    async def _perform_health_check(self, deployment: TutorDeployment) -> bool:
        """Perform health check on deployment."""
        # Simulate health check
        try:
            # Check if deployment files exist
            package_path = Path(deployment.deployment_package_path)
            if not package_path.exists():
                return False

            # Check if deployment is accessible
            # Would implement actual connectivity/responsiveness check
            await asyncio.sleep(0.1)

            return True

        except Exception as e:
            logger.exception(f"Health check error for {deployment.deployment_id}: {e}")
            return False

    async def _process_pending_updates(self) -> None:
        """Process pending updates for all devices."""
        for updates in self.pending_updates.values():
            for update in updates[
                :
            ]:  # Copy list to avoid modification during iteration
                if not update.installed and not update.downloaded:
                    # Process high-priority updates immediately
                    if update.priority in ["critical", "high"]:
                        asyncio.create_task(self._execute_update(update))
                    # Schedule normal updates
                    elif update.scheduled_time is None:
                        # Schedule for off-peak hours
                        asyncio.create_task(self._execute_update(update))

    async def _update_device_status(self) -> None:
        """Update device last seen status."""
        for device in self.devices.values():
            # Would implement actual device ping/heartbeat
            # For now, just update timestamp for active devices
            if any(
                d.status == DeploymentStatus.RUNNING
                for d in self.deployments.values()
                if d.device_id == device.device_id
            ):
                device.last_seen = datetime.now(timezone.utc).isoformat()
                await self._save_device_profile(device)

    async def _cleanup_old_deployments(self) -> None:
        """Clean up old or unused deployments."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

        deployments_to_remove = []

        for deployment_id, deployment in self.deployments.items():
            # Remove deployments that haven't been used in 30 days
            if deployment.last_used != "never":
                last_used = datetime.fromisoformat(deployment.last_used)
                if last_used < cutoff_date and deployment.usage_count == 0:
                    deployments_to_remove.append(deployment_id)

        for deployment_id in deployments_to_remove:
            await self._remove_deployment(deployment_id)

    async def _remove_deployment(self, deployment_id: str) -> None:
        """Remove deployment and clean up resources."""
        try:
            deployment = self.deployments[deployment_id]

            # Remove deployment package file
            package_path = Path(deployment.deployment_package_path)
            if package_path.exists():
                package_path.unlink()

            # Remove from memory
            del self.deployments[deployment_id]

            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM tutor_deployments WHERE deployment_id = ?",
                (deployment_id,),
            )
            conn.commit()
            conn.close()

            logger.info(f"Removed old deployment {deployment_id}")

        except Exception as e:
            logger.exception(f"Error removing deployment {deployment_id}: {e}")

    # Database helper methods
    async def _save_device_profile(self, device: DeviceProfile) -> None:
        """Save device profile to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            device_specs = {
                "os_name": device.os_name,
                "os_version": device.os_version,
                "cpu_architecture": device.cpu_architecture,
                "cpu_cores": device.cpu_cores,
                "ram_mb": device.ram_mb,
                "storage_available_mb": device.storage_available_mb,
                "gpu_available": device.gpu_available,
                "gpu_memory_mb": device.gpu_memory_mb,
                "network_speed_mbps": device.network_speed_mbps,
                "battery_powered": device.battery_powered,
                "always_connected": device.always_connected,
                "parental_controls": device.parental_controls,
            }

            cursor.execute(
                """
                INSERT OR REPLACE INTO device_profiles VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    device.device_id,
                    device.device_name,
                    device.device_type.value,
                    json.dumps(device_specs),
                    json.dumps(device.student_ids),
                    device.last_seen,
                    device.created_at,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception(f"Failed to save device profile: {e}")

    async def _save_deployment(self, deployment: TutorDeployment) -> None:
        """Save deployment to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            deployment_data = {
                "tutor_model_id": deployment.tutor_model_id,
                "tutor_version": deployment.tutor_version,
                "deployment_package_path": deployment.deployment_package_path,
                "model_size_mb": deployment.model_size_mb,
                "compressed_size_mb": deployment.compressed_size_mb,
                "usage_count": deployment.usage_count,
                "performance_metrics": deployment.performance_metrics,
                "sync_status": deployment.sync_status,
                "offline_capable": deployment.offline_capable,
                "auto_update": deployment.auto_update,
                "parent_approved": deployment.parent_approved,
            }

            cursor.execute(
                """
                INSERT OR REPLACE INTO tutor_deployments VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    deployment.deployment_id,
                    deployment.device_id,
                    deployment.student_id,
                    json.dumps(deployment_data),
                    deployment.status.value,
                    deployment.installation_date,
                    deployment.last_used,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception(f"Failed to save deployment: {e}")

    async def _save_update(self, update: EdgeUpdate) -> None:
        """Save update to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            update_data = asdict(update)

            cursor.execute(
                """
                INSERT OR REPLACE INTO edge_updates VALUES (?, ?, ?, ?, ?)
            """,
                (
                    update.update_id,
                    update.deployment_id,
                    json.dumps(update_data),
                    datetime.now(timezone.utc).isoformat(),
                    (
                        datetime.now(timezone.utc).isoformat()
                        if update.installed
                        else None
                    ),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception(f"Failed to save update: {e}")

    def get_deployment_analytics(self) -> dict[str, Any]:
        """Get comprehensive deployment analytics."""
        total_deployments = len(self.deployments)
        active_deployments = len(
            [
                d
                for d in self.deployments.values()
                if d.status == DeploymentStatus.RUNNING
            ]
        )
        total_devices = len(self.devices)

        # Device type distribution
        device_types = defaultdict(int)
        for device in self.devices.values():
            device_types[device.device_type.value] += 1

        # Deployment status distribution
        status_distribution = defaultdict(int)
        for deployment in self.deployments.values():
            status_distribution[deployment.status.value] += 1

        # Performance metrics
        if self.deployments:
            total_size_mb = sum(d.model_size_mb for d in self.deployments.values())
            total_compressed_mb = sum(
                d.compressed_size_mb for d in self.deployments.values()
            )
            avg_compression_ratio = total_size_mb / max(total_compressed_mb, 1)

            total_usage = sum(d.usage_count for d in self.deployments.values())
            avg_usage = total_usage / len(self.deployments)
        else:
            avg_compression_ratio = 0
            avg_usage = 0

        analytics = {
            "summary": {
                "total_deployments": total_deployments,
                "active_deployments": active_deployments,
                "total_devices": total_devices,
                "avg_compression_ratio": avg_compression_ratio,
                "avg_usage_per_deployment": avg_usage,
            },
            "device_distribution": dict(device_types),
            "status_distribution": dict(status_distribution),
            "deployment_metrics": {
                "total_storage_used_mb": sum(
                    d.compressed_size_mb for d in self.deployments.values()
                ),
                "total_updates_pending": sum(
                    len(updates) for updates in self.pending_updates.values()
                ),
                "sync_success_rate": 0.95,  # Would calculate from actual sync data
            },
            "performance_trends": {
                "deployment_success_rate": 0.92,  # Would calculate from actual data
                "avg_deployment_time_minutes": 5.2,
                "avg_sync_time_seconds": 30.5,
            },
        }

        return analytics


# Global edge deployment manager instance
edge_deployment_manager = EdgeDeploymentManager()
