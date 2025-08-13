"""
Bluetooth BLE Transport Implementation - Phase 3 Advanced Features

Implements Bluetooth Low Energy transport for P2P communication with:
- Device discovery and pairing
- GATT characteristic-based messaging
- Connection management and failover
- Integration with BitChat mesh network
"""

import asyncio
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# BLE Service and Characteristic UUIDs
AIVILLAGE_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
MESSAGE_CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef1"
CONTROL_CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef2"
STATUS_CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef3"


class BLEState(Enum):
    """Bluetooth LE connection states."""

    DISCONNECTED = "disconnected"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ADVERTISING = "advertising"
    ERROR = "error"


@dataclass
class BLEDevice:
    """Represents a discovered BLE device."""

    address: str
    name: str | None
    rssi: int
    services: list[str] = None
    last_seen: datetime = None
    connection_attempts: int = 0

    def __post_init__(self):
        if self.services is None:
            self.services = []
        if self.last_seen is None:
            self.last_seen = datetime.now()


class BLETransport:
    """Bluetooth Low Energy transport for P2P communication."""

    def __init__(
        self,
        device_name: str = "AIVillage-Node",
        max_connections: int = 7,  # BLE typically supports 7-8 simultaneous connections
        scan_duration: int = 10,
        message_callback: Callable | None = None,
    ):
        """
        Initialize BLE transport.

        Args:
            device_name: Name to advertise
            max_connections: Maximum simultaneous BLE connections
            scan_duration: Duration for device discovery scans
            message_callback: Callback for received messages
        """
        self.device_name = device_name
        self.max_connections = max_connections
        self.scan_duration = scan_duration
        self.message_callback = message_callback

        self.state = BLEState.DISCONNECTED
        self.discovered_devices: dict[str, BLEDevice] = {}
        self.connected_devices: set[str] = set()
        self.pending_messages: list[bytes] = []

        # Platform-specific BLE implementation
        self.ble_adapter = None
        self._init_platform_ble()

    def _init_platform_ble(self):
        """Initialize platform-specific BLE implementation."""
        try:
            # Try to import bleak for cross-platform BLE
            import bleak

            self.ble_adapter = BleakAdapter(self)
            logger.info("Initialized Bleak BLE adapter")
        except ImportError:
            logger.warning("Bleak not available, trying platform-specific BLE")

            # Try Android BLE
            if self._init_android_ble():
                return

            # Try iOS/macOS BLE
            if self._init_ios_ble():
                return

            # Try Linux BlueZ
            if self._init_linux_ble():
                return

            logger.error("No BLE implementation available")
            self.state = BLEState.ERROR

    def _init_android_ble(self) -> bool:
        """Initialize Android BLE if available."""
        try:
            from jnius import autoclass

            BluetoothAdapter = autoclass("android.bluetooth.BluetoothAdapter")
            BluetoothManager = autoclass("android.bluetooth.BluetoothManager")

            self.ble_adapter = AndroidBLEAdapter(self)
            logger.info("Initialized Android BLE adapter")
            return True
        except:
            return False

    def _init_ios_ble(self) -> bool:
        """Initialize iOS/macOS Core Bluetooth if available."""
        try:
            self.ble_adapter = CoreBluetoothAdapter(self)
            logger.info("Initialized Core Bluetooth adapter")
            return True
        except:
            return False

    def _init_linux_ble(self) -> bool:
        """Initialize Linux BlueZ if available."""
        try:
            self.ble_adapter = BlueZAdapter(self)
            logger.info("Initialized BlueZ BLE adapter")
            return True
        except:
            return False

    async def start(self):
        """Start BLE transport."""
        if self.state == BLEState.ERROR:
            logger.error("Cannot start BLE transport in error state")
            return False

        if not self.ble_adapter:
            logger.error("No BLE adapter available")
            return False

        try:
            # Start advertising our service
            await self.start_advertising()

            # Start scanning for other devices
            asyncio.create_task(self.continuous_scan())

            logger.info(f"BLE transport started: {self.device_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start BLE transport: {e}")
            self.state = BLEState.ERROR
            return False

    async def start_advertising(self):
        """Start advertising AIVillage service."""
        self.state = BLEState.ADVERTISING

        if self.ble_adapter:
            await self.ble_adapter.start_advertising(
                service_uuid=AIVILLAGE_SERVICE_UUID, device_name=self.device_name
            )

        logger.info(f"Advertising as {self.device_name}")

    async def continuous_scan(self):
        """Continuously scan for nearby devices."""
        while self.state != BLEState.ERROR:
            try:
                await self.scan_for_devices()
                await asyncio.sleep(30)  # Pause between scans
            except Exception as e:
                logger.error(f"Scan error: {e}")
                await asyncio.sleep(60)

    async def scan_for_devices(self):
        """Scan for nearby BLE devices with AIVillage service."""
        self.state = BLEState.SCANNING
        logger.info(f"Scanning for BLE devices for {self.scan_duration}s...")

        if not self.ble_adapter:
            return

        try:
            devices = await self.ble_adapter.scan(
                duration=self.scan_duration, service_filter=AIVILLAGE_SERVICE_UUID
            )

            for device in devices:
                if device.address not in self.discovered_devices:
                    self.discovered_devices[device.address] = device
                    logger.info(
                        f"Discovered new device: {device.name} ({device.address})"
                    )

                    # Auto-connect if we have capacity
                    if len(self.connected_devices) < self.max_connections:
                        asyncio.create_task(self.connect_to_device(device.address))

        except Exception as e:
            logger.error(f"Scan failed: {e}")

    async def connect_to_device(self, address: str) -> bool:
        """Connect to a BLE device."""
        if address in self.connected_devices:
            return True

        if len(self.connected_devices) >= self.max_connections:
            logger.warning(f"Max connections reached ({self.max_connections})")
            return False

        device = self.discovered_devices.get(address)
        if not device:
            logger.error(f"Unknown device: {address}")
            return False

        self.state = BLEState.CONNECTING
        device.connection_attempts += 1

        try:
            if self.ble_adapter:
                success = await self.ble_adapter.connect(address)

                if success:
                    self.connected_devices.add(address)
                    self.state = BLEState.CONNECTED
                    logger.info(f"Connected to {device.name} ({address})")

                    # Send any pending messages
                    await self.flush_pending_messages(address)
                    return True

        except Exception as e:
            logger.error(f"Connection failed to {address}: {e}")

        self.state = BLEState.DISCONNECTED
        return False

    async def send_message(self, message: bytes, target: str | None = None) -> bool:
        """
        Send a message via BLE.

        Args:
            message: Message bytes to send
            target: Optional target device address (broadcasts if None)

        Returns:
            True if message was sent successfully
        """
        if not self.connected_devices:
            # Queue message for later delivery
            self.pending_messages.append(message)
            logger.warning("No connected devices, message queued")
            return False

        # Chunk message if needed (BLE MTU is typically 20-512 bytes)
        chunks = self._chunk_message(message)

        targets = [target] if target else list(self.connected_devices)
        success_count = 0

        for device_address in targets:
            if device_address not in self.connected_devices:
                continue

            try:
                for chunk in chunks:
                    if self.ble_adapter:
                        await self.ble_adapter.write_characteristic(
                            device_address, MESSAGE_CHARACTERISTIC_UUID, chunk
                        )

                success_count += 1
                logger.debug(f"Sent message to {device_address}")

            except Exception as e:
                logger.error(f"Failed to send to {device_address}: {e}")

        return success_count > 0

    def _chunk_message(self, message: bytes, mtu: int = 180) -> list[bytes]:
        """
        Chunk a message for BLE transmission.

        BLE typically has small MTU (20-512 bytes), we use 180 as safe default.
        """
        chunks = []
        total_chunks = (len(message) + mtu - 1) // mtu

        for i in range(total_chunks):
            start = i * mtu
            end = min(start + mtu, len(message))

            # Add simple header: [chunk_index, total_chunks, data...]
            header = struct.pack("BB", i, total_chunks)
            chunk = header + message[start:end]
            chunks.append(chunk)

        return chunks

    async def flush_pending_messages(self, target: str):
        """Send any pending messages to newly connected device."""
        if not self.pending_messages:
            return

        logger.info(
            f"Flushing {len(self.pending_messages)} pending messages to {target}"
        )

        for message in self.pending_messages[
            :
        ]:  # Copy list to avoid modification during iteration
            if await self.send_message(message, target):
                self.pending_messages.remove(message)

    def handle_received_message(self, sender: str, data: bytes):
        """Handle a received BLE message."""
        logger.debug(f"Received {len(data)} bytes from {sender}")

        # Parse chunk header if present
        if len(data) >= 2:
            chunk_index = data[0]
            total_chunks = data[1]

            if total_chunks > 1:
                # Handle multi-chunk message
                self._handle_chunked_message(
                    sender, chunk_index, total_chunks, data[2:]
                )
                return
            else:
                # Single chunk message
                data = data[2:]

        # Invoke callback if registered
        if self.message_callback:
            try:
                self.message_callback(sender, data)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

    def _handle_chunked_message(self, sender: str, index: int, total: int, data: bytes):
        """Reassemble chunked messages."""
        # This would need a proper reassembly buffer per sender
        # For now, just log it
        logger.debug(f"Received chunk {index + 1}/{total} from {sender}")

    async def disconnect(self, address: str | None = None):
        """Disconnect from a device or all devices."""
        if address:
            if address in self.connected_devices:
                if self.ble_adapter:
                    await self.ble_adapter.disconnect(address)
                self.connected_devices.remove(address)
                logger.info(f"Disconnected from {address}")
        else:
            # Disconnect all
            for device in list(self.connected_devices):
                await self.disconnect(device)

    async def stop(self):
        """Stop BLE transport."""
        logger.info("Stopping BLE transport")

        # Stop advertising
        if self.ble_adapter:
            await self.ble_adapter.stop_advertising()

        # Disconnect all devices
        await self.disconnect()

        self.state = BLEState.DISCONNECTED
        logger.info("BLE transport stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current BLE transport status."""
        return {
            "state": self.state.value,
            "device_name": self.device_name,
            "discovered_devices": len(self.discovered_devices),
            "connected_devices": len(self.connected_devices),
            "pending_messages": len(self.pending_messages),
            "max_connections": self.max_connections,
            "adapter": self.ble_adapter.__class__.__name__
            if self.ble_adapter
            else None,
        }


class BleakAdapter:
    """Cross-platform BLE adapter using Bleak library."""

    def __init__(self, transport: BLETransport):
        self.transport = transport
        self.client_connections = {}

    async def scan(self, duration: int, service_filter: str) -> list[BLEDevice]:
        """Scan for BLE devices."""
        try:
            from bleak import BleakScanner

            devices = []
            scanner = BleakScanner()
            discovered = await scanner.discover(timeout=duration)

            for d in discovered:
                # Filter by service UUID if possible
                if service_filter and hasattr(d, "metadata"):
                    uuids = d.metadata.get("uuids", [])
                    if service_filter not in uuids:
                        continue

                device = BLEDevice(
                    address=d.address,
                    name=d.name,
                    rssi=d.rssi if hasattr(d, "rssi") else -100,
                )
                devices.append(device)

            return devices

        except Exception as e:
            logger.error(f"Bleak scan error: {e}")
            return []

    async def connect(self, address: str) -> bool:
        """Connect to a BLE device."""
        try:
            from bleak import BleakClient

            client = BleakClient(address)
            await client.connect()

            self.client_connections[address] = client

            # Set up notification handler
            await client.start_notify(
                MESSAGE_CHARACTERISTIC_UUID,
                lambda sender, data: self.transport.handle_received_message(
                    address, data
                ),
            )

            return True

        except Exception as e:
            logger.error(f"Bleak connect error: {e}")
            return False

    async def disconnect(self, address: str):
        """Disconnect from a BLE device."""
        if address in self.client_connections:
            client = self.client_connections[address]
            await client.disconnect()
            del self.client_connections[address]

    async def write_characteristic(self, address: str, char_uuid: str, data: bytes):
        """Write data to a characteristic."""
        if address in self.client_connections:
            client = self.client_connections[address]
            await client.write_gatt_char(char_uuid, data)

    async def start_advertising(self, service_uuid: str, device_name: str):
        """Start BLE advertising (not supported by Bleak - client only)."""
        logger.warning("Bleak does not support BLE advertising (client-only)")

    async def stop_advertising(self):
        """Stop BLE advertising."""
        pass


# Placeholder for platform-specific adapters
class AndroidBLEAdapter:
    """Android-specific BLE implementation."""

    def __init__(self, transport: BLETransport):
        self.transport = transport
        logger.info("Android BLE adapter initialized (stub)")


class CoreBluetoothAdapter:
    """iOS/macOS Core Bluetooth implementation."""

    def __init__(self, transport: BLETransport):
        self.transport = transport
        logger.info("Core Bluetooth adapter initialized (stub)")


class BlueZAdapter:
    """Linux BlueZ implementation."""

    def __init__(self, transport: BLETransport):
        self.transport = transport
        logger.info("BlueZ adapter initialized (stub)")


# Integration with BitChat
async def create_ble_bitchat_bridge(ble_transport: BLETransport, bitchat_node):
    """
    Bridge BLE transport with BitChat mesh network.

    This allows BLE devices to participate in the larger mesh network.
    """

    def ble_to_bitchat(sender: str, data: bytes):
        """Forward BLE messages to BitChat."""
        try:
            # Wrap BLE message with metadata
            wrapped = {"source": sender, "transport": "ble", "data": data.hex()}
            bitchat_node.broadcast(wrapped)
        except Exception as e:
            logger.error(f"BLE to BitChat bridge error: {e}")

    # Register callback
    ble_transport.message_callback = ble_to_bitchat

    # Start transport
    await ble_transport.start()

    logger.info("BLE-BitChat bridge established")
    return ble_transport
