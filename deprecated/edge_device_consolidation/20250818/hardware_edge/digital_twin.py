"""
Personal AI Digital Twin Concierge

Runs locally on edge devices (phones, IoT) and:
- Absorbs user data for personalized training
- Handles simple tasks locally
- Manages tiny sharded model training
- Coordinates with fog cloud for complex tasks
- Sells unused compute during night charging
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    IOT_DEVICE = "iot_device"
    EDGE_SERVER = "edge_server"


class ComputeMode(Enum):
    PERSONAL_USE = "personal_use"  # Day time, user tasks
    FOG_CONTRIBUTION = "fog_contribution"  # Night time, selling compute
    TRAINING = "training"  # Local model training
    STANDBY = "standby"  # Low power mode


@dataclass
class DeviceResources:
    """Current device resource status"""

    battery_percent: float
    cpu_usage: float
    memory_usage: float
    storage_available_gb: float
    network_type: str  # "wifi", "cellular", "offline"
    temperature_celsius: float
    charging: bool


@dataclass
class UserInteraction:
    """User interaction data for personalization"""

    timestamp: float
    interaction_type: str  # "query", "preference", "feedback"
    content: str
    context: dict[str, Any]
    response: str | None = None


class DigitalTwinConcierge:
    """
    Personal AI concierge running on each edge device

    Core Functions:
    1. Local task handling (weather, reminders, simple queries)
    2. User data absorption and model personalization
    3. Resource management (battery, compute, network)
    4. Fog cloud coordination (selling compute at night)
    5. Privacy-first data handling
    """

    def __init__(self, device_id: str, device_type: DeviceType, user_id: str):
        self.device_id = device_id
        self.device_type = device_type
        self.user_id = user_id

        # Local AI model (tiny, sharded)
        self.local_model = None
        self.model_version = "v1.0"
        self.personalization_data = []

        # Device state
        self.current_mode = ComputeMode.PERSONAL_USE
        self.resources = DeviceResources(
            battery_percent=100.0,
            cpu_usage=0.1,
            memory_usage=0.3,
            storage_available_gb=50.0,
            network_type="wifi",
            temperature_celsius=25.0,
            charging=False,
        )

        # Communication with broader system
        self.bitchat_transport = None
        self.betanet_transport = None

        # Task handling
        self.local_capabilities = [
            "weather",
            "time",
            "calendar",
            "reminders",
            "notes",
            "calculator",
            "timer",
            "basic_qa",
            "preferences",
        ]

        # Fog cloud participation
        self.fog_earnings_credits = 0.0
        self.compute_contribution_hours = 0.0

        self.initialized = False

    async def initialize(self):
        """Initialize the digital twin concierge"""
        try:
            logger.info(
                f"Initializing Digital Twin on {self.device_type.value} device {self.device_id}"
            )

            # Initialize local AI model
            await self._initialize_local_model()

            # Setup communication protocols
            await self._setup_communication()

            # Load personalization data
            await self._load_personalization_data()

            # Start background tasks
            asyncio.create_task(self._resource_monitor())
            asyncio.create_task(self._fog_cloud_scheduler())
            asyncio.create_task(self._model_training_scheduler())

            self.initialized = True
            logger.info(f"✅ Digital Twin {self.device_id} initialized")

        except Exception as e:
            logger.error(f"❌ Digital Twin initialization failed: {e}")
            raise

    async def process_user_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Process user request - first entry point in AIVillage system

        Decision Flow:
        1. Can I handle this locally? -> Handle and respond
        2. Is this complex? -> Route to King Agent via BitChat/BetaNet
        3. Need new capability? -> Request Agent Forge
        """
        request_id = f"req_{self.device_id}_{int(time.time())}"

        # Log user interaction for personalization
        interaction = UserInteraction(
            timestamp=time.time(),
            interaction_type="query",
            content=request.get("query", ""),
            context=request.get("context", {}),
        )

        # Analyze if we can handle locally
        can_handle_local = await self._can_handle_locally(request)

        if can_handle_local:
            # Handle locally for speed and privacy
            response = await self._handle_local_request(request)
            interaction.response = response.get("content", "")

            # Learn from interaction
            await self._absorb_interaction(interaction)

            return {
                "request_id": request_id,
                "handler": "digital_twin_local",
                "response": response,
                "processing_time_ms": response.get("processing_time_ms", 50),
                "privacy": "local_only",
            }
        else:
            # Route to meta-agent system
            return await self._route_to_meta_agents(request, request_id)

    async def _can_handle_locally(self, request: dict[str, Any]) -> bool:
        """Determine if request can be handled locally"""
        query = request.get("query", "").lower()

        # Check against local capabilities
        for capability in self.local_capabilities:
            if capability in query:
                return True

        # Check for personalized patterns the local model learned
        if await self._matches_learned_patterns(query):
            return True

        return False

    async def _handle_local_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle request using local AI model and capabilities"""
        start_time = time.time()
        query = request.get("query", "")

        # Simple local responses (would be handled by tiny local model)
        local_responses = {
            "weather": f"Current weather for {self.user_id}: Sunny, 72°F",
            "time": f"Current time: {time.strftime('%H:%M %Z')}",
            "battery": f"Device battery: {self.resources.battery_percent:.0f}%",
            "reminder": "I've added that reminder to your personal schedule",
            "note": "Note saved to your private device storage",
        }

        # Find best matching response
        response_content = "I've processed your request locally."
        for keyword, response in local_responses.items():
            if keyword in query.lower():
                response_content = response
                break

        processing_time = (time.time() - start_time) * 1000  # ms

        return {
            "content": response_content,
            "source": "local_digital_twin",
            "personalized": True,
            "processing_time_ms": processing_time,
            "model_version": self.model_version,
        }

    async def _route_to_meta_agents(
        self, request: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        """Route complex requests to meta-agent system"""
        logger.info(f"Routing request {request_id} to meta-agent system")

        # Choose communication protocol based on network/privacy needs
        if (
            self.resources.network_type == "offline"
            or request.get("privacy_level") == "high"
        ):
            transport = "bitchat"  # Use Bluetooth mesh
        else:
            transport = "betanet"  # Use encrypted internet

        # Send to architectural orchestrator
        meta_request = {
            "request_id": request_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "description": request.get("query", ""),
            "context": request.get("context", {}),
            "requirements": {
                "privacy_level": request.get("privacy_level", "standard"),
                "urgency": request.get("urgency", "normal"),
                "capabilities": request.get("capabilities", []),
            },
            "transport_used": transport,
        }

        return {
            "request_id": request_id,
            "handler": "routing_to_meta_agents",
            "transport": transport,
            "status": "delegated",
            "meta_request": meta_request,
        }

    async def _absorb_interaction(self, interaction: UserInteraction):
        """Absorb user interaction for model personalization"""
        self.personalization_data.append(interaction)

        # Trigger local model retraining if enough new data
        if len(self.personalization_data) % 10 == 0:  # Every 10 interactions
            asyncio.create_task(self._retrain_local_model())

    async def _matches_learned_patterns(self, query: str) -> bool:
        """Check if query matches patterns the local model learned"""
        # Would use actual local AI model inference
        return False  # Placeholder

    async def _initialize_local_model(self):
        """Initialize tiny local AI model for personal tasks"""
        logger.info(f"Loading local AI model for device {self.device_id}")
        # Would load actual compressed model (BitNet, quantized)
        self.local_model = "tiny_personal_model_v1.0"  # Placeholder

    async def _setup_communication(self):
        """Setup BitChat and BetaNet communication"""
        # Would initialize actual communication protocols
        self.bitchat_transport = "bluetooth_mesh_ready"
        self.betanet_transport = "encrypted_internet_ready"

    async def _load_personalization_data(self):
        """Load existing personalization data from secure local storage"""
        # Would load from encrypted local database
        self.personalization_data = []  # Placeholder

    async def _resource_monitor(self):
        """Monitor device resources and adjust behavior"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Update resource status (would read from actual device APIs)
                await self._update_resource_status()

                # Adjust compute mode based on resources
                await self._adjust_compute_mode()

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    async def _fog_cloud_scheduler(self):
        """Schedule fog cloud participation during optimal times"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check if conditions are right for fog participation
                if await self._should_join_fog_cloud():
                    await self._start_fog_contribution()
                elif self.current_mode == ComputeMode.FOG_CONTRIBUTION:
                    await self._stop_fog_contribution()

            except Exception as e:
                logger.error(f"Fog cloud scheduling error: {e}")

    async def _model_training_scheduler(self):
        """Schedule local model training during optimal times"""
        while True:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes

                if await self._should_train_model():
                    await self._retrain_local_model()

            except Exception as e:
                logger.error(f"Model training scheduling error: {e}")

    async def _should_join_fog_cloud(self) -> bool:
        """Determine if device should contribute to fog cloud"""
        # Join fog cloud at night while charging with good battery
        current_hour = int(time.strftime("%H"))
        is_night = current_hour >= 22 or current_hour <= 6

        return (
            is_night
            and self.resources.charging
            and self.resources.battery_percent > 50
            and self.resources.cpu_usage < 0.3
            and self.resources.temperature_celsius < 40
            and self.resources.network_type != "offline"
        )

    async def _start_fog_contribution(self):
        """Start contributing compute to fog cloud"""
        if self.current_mode != ComputeMode.FOG_CONTRIBUTION:
            logger.info(f"Device {self.device_id} joining fog cloud")
            self.current_mode = ComputeMode.FOG_CONTRIBUTION
            # Would register with fog cloud coordinator

    async def _stop_fog_contribution(self):
        """Stop contributing compute to fog cloud"""
        logger.info(f"Device {self.device_id} leaving fog cloud")
        self.current_mode = ComputeMode.PERSONAL_USE
        # Would deregister from fog cloud coordinator

    async def _should_train_model(self) -> bool:
        """Determine if local model should be retrained"""
        return (
            len(self.personalization_data) >= 10
            and self.resources.charging
            and self.resources.battery_percent > 80
            and self.current_mode != ComputeMode.FOG_CONTRIBUTION
        )

    async def _retrain_local_model(self):
        """Retrain local model with new personalization data"""
        if self.current_mode != ComputeMode.TRAINING:
            logger.info(f"Retraining local model on device {self.device_id}")
            old_mode = self.current_mode
            self.current_mode = ComputeMode.TRAINING

            try:
                # Would perform actual local training
                await asyncio.sleep(2)  # Simulate training time
                self.model_version = f"v{float(self.model_version[1:]) + 0.1:.1f}"
                logger.info(f"Local model updated to {self.model_version}")

            finally:
                self.current_mode = old_mode

    async def _update_resource_status(self):
        """Update current device resource status"""
        # Would read from actual device APIs - simulated for now
        pass

    async def _adjust_compute_mode(self):
        """Adjust compute mode based on current resources"""
        if self.resources.battery_percent < 20:
            self.current_mode = ComputeMode.STANDBY
        elif self.resources.temperature_celsius > 50:
            self.current_mode = ComputeMode.STANDBY

    async def get_status(self) -> dict[str, Any]:
        """Get current digital twin status"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "user_id": self.user_id,
            "current_mode": self.current_mode.value,
            "model_version": self.model_version,
            "personalization_interactions": len(self.personalization_data),
            "fog_earnings_credits": self.fog_earnings_credits,
            "fog_contribution_hours": self.compute_contribution_hours,
            "resources": {
                "battery_percent": self.resources.battery_percent,
                "charging": self.resources.charging,
                "network_type": self.resources.network_type,
                "cpu_usage": self.resources.cpu_usage,
                "memory_usage": self.resources.memory_usage,
            },
            "local_capabilities": self.local_capabilities,
            "communication": {
                "bitchat": self.bitchat_transport is not None,
                "betanet": self.betanet_transport is not None,
            },
        }
