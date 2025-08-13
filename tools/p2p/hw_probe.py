#!/usr/bin/env python3
"""Hardware Probe for BitChat/Betanet P2P Networking

Graceful hardware detection and capability testing for:
- Bluetooth Low Energy (BLE) availability
- WiFi adapter capabilities
- Network interface detection
- P2P transport readiness

Designed to timeout gracefully when hardware is not present, printing:
- "HARDWARE_OK" when hardware is available and functional
- "SKIPPED" when hardware is not present (without crashing)

Usage:
    python tools/p2p/hw_probe.py
    python tools/p2p/hw_probe.py || true  # Safe execution
"""

import asyncio
import json
import logging
import platform
import sys
import time
from typing import Any

# Configure logging to suppress unnecessary output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class HardwareProbe:
    """Hardware capability probe with graceful timeout handling"""

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        self.results: dict[str, Any] = {
            "bluetooth": {"available": False, "details": {}},
            "wifi": {"available": False, "details": {}},
            "network": {"interfaces": [], "p2p_ready": False},
            "platform": {
                "system": platform.system(),
                "version": platform.version(),
                "architecture": platform.architecture()[0],
            },
            "summary": {
                "hardware_ok": False,
                "reason": "Not tested",
                "recommendations": [],
            },
        }

    async def probe_all_hardware(self) -> dict[str, Any]:
        """Probe all hardware with timeout protection"""
        try:
            # Run all probes concurrently with timeout
            await asyncio.wait_for(self._run_all_probes(), timeout=self.timeout_seconds)

        except asyncio.TimeoutError:
            logger.warning(f"Hardware probe timed out after {self.timeout_seconds}s")
            self.results["summary"]["reason"] = "Probe timeout"

        except Exception as e:
            logger.warning(f"Hardware probe failed: {e}")
            self.results["summary"]["reason"] = f"Probe error: {e!s}"

        # Determine overall hardware status
        self._determine_hardware_status()
        return self.results

    async def _run_all_probes(self):
        """Run all hardware probes concurrently"""
        await asyncio.gather(
            self._probe_bluetooth(),
            self._probe_wifi(),
            self._probe_network_interfaces(),
            return_exceptions=True,
        )

    async def _probe_bluetooth(self):
        """Probe Bluetooth/BLE capabilities with timeout"""
        try:
            # Test 1: Check for PyBluez library
            try:
                import bluetooth

                self.results["bluetooth"]["pybluez_available"] = True
                logger.debug("PyBluez library available")
            except ImportError:
                self.results["bluetooth"]["pybluez_available"] = False
                logger.debug("PyBluez library not available")

            # Test 2: Platform-specific Bluetooth detection
            if platform.system() == "Linux":
                await self._probe_bluetooth_linux()
            elif platform.system() == "Windows":
                await self._probe_bluetooth_windows()
            elif platform.system() == "Darwin":  # macOS
                await self._probe_bluetooth_macos()
            else:
                self.results["bluetooth"]["details"]["error"] = "Unsupported platform"

            # Test 3: Try actual Bluetooth discovery (quick test)
            if self.results["bluetooth"].get("adapter_present", False):
                await self._test_bluetooth_discovery()

        except Exception as e:
            logger.debug(f"Bluetooth probe failed: {e}")
            self.results["bluetooth"]["error"] = str(e)

    async def _probe_bluetooth_linux(self):
        """Linux-specific Bluetooth detection"""
        try:
            # Check for Bluetooth adapters via hciconfig
            result = await self._run_command_timeout(["hciconfig"], timeout=3)
            if result and "hci" in result.lower():
                self.results["bluetooth"]["adapter_present"] = True
                self.results["bluetooth"]["details"]["hciconfig"] = result[:200]
            else:
                self.results["bluetooth"]["adapter_present"] = False

            # Check bluetoothctl
            result = await self._run_command_timeout(
                ["bluetoothctl", "--version"], timeout=2
            )
            if result:
                self.results["bluetooth"]["details"]["bluetoothctl"] = result.strip()

        except Exception as e:
            logger.debug(f"Linux Bluetooth probe failed: {e}")

    async def _probe_bluetooth_windows(self):
        """Windows-specific Bluetooth detection"""
        try:
            # Use PowerShell to check Bluetooth adapters
            powershell_cmd = [
                "powershell",
                "-Command",
                "Get-PnpDevice -Class Bluetooth -Status OK | Select-Object FriendlyName, Status",
            ]

            result = await self._run_command_timeout(powershell_cmd, timeout=5)
            if result and "bluetooth" in result.lower():
                self.results["bluetooth"]["adapter_present"] = True
                self.results["bluetooth"]["details"]["adapters"] = result[:300]
            else:
                self.results["bluetooth"]["adapter_present"] = False

            # Check Windows Bluetooth service
            service_cmd = [
                "powershell",
                "-Command",
                'Get-Service -Name "bthserv" | Select-Object Status',
            ]

            service_result = await self._run_command_timeout(service_cmd, timeout=3)
            if service_result:
                self.results["bluetooth"]["details"]["service_status"] = (
                    service_result.strip()
                )

        except Exception as e:
            logger.debug(f"Windows Bluetooth probe failed: {e}")

    async def _probe_bluetooth_macos(self):
        """macOS-specific Bluetooth detection"""
        try:
            # Use system_profiler to check Bluetooth
            result = await self._run_command_timeout(
                ["system_profiler", "SPBluetoothDataType"], timeout=5
            )

            if result and "bluetooth" in result.lower():
                self.results["bluetooth"]["adapter_present"] = True
                self.results["bluetooth"]["details"]["system_profiler"] = result[:300]
            else:
                self.results["bluetooth"]["adapter_present"] = False

        except Exception as e:
            logger.debug(f"macOS Bluetooth probe failed: {e}")

    async def _test_bluetooth_discovery(self):
        """Test actual Bluetooth device discovery"""
        try:
            # Only test if PyBluez is available
            if not self.results["bluetooth"].get("pybluez_available", False):
                return

            import bluetooth

            # Quick discovery test (2 second timeout)
            start_time = time.time()
            devices = bluetooth.discover_devices(duration=2, lookup_names=False)
            discovery_time = time.time() - start_time

            self.results["bluetooth"]["discovery_test"] = {
                "devices_found": len(devices),
                "discovery_time_seconds": round(discovery_time, 2),
                "discovery_working": True,
            }

            self.results["bluetooth"]["available"] = True
            logger.debug(f"Bluetooth discovery found {len(devices)} devices")

        except Exception as e:
            logger.debug(f"Bluetooth discovery test failed: {e}")
            self.results["bluetooth"]["discovery_test"] = {
                "error": str(e),
                "discovery_working": False,
            }

    async def _probe_wifi(self):
        """Probe WiFi capabilities"""
        try:
            if platform.system() == "Linux":
                await self._probe_wifi_linux()
            elif platform.system() == "Windows":
                await self._probe_wifi_windows()
            elif platform.system() == "Darwin":  # macOS
                await self._probe_wifi_macos()

        except Exception as e:
            logger.debug(f"WiFi probe failed: {e}")
            self.results["wifi"]["error"] = str(e)

    async def _probe_wifi_linux(self):
        """Linux WiFi detection"""
        try:
            # Check for wireless interfaces
            result = await self._run_command_timeout(["iwconfig"], timeout=3)
            if result:
                wireless_interfaces = []
                for line in result.split("\n"):
                    if "IEEE 802.11" in line:
                        interface = line.split()[0]
                        wireless_interfaces.append(interface)

                self.results["wifi"]["interfaces"] = wireless_interfaces
                self.results["wifi"]["available"] = len(wireless_interfaces) > 0

            # Check WiFi Direct/P2P support
            p2p_result = await self._run_command_timeout(["iw", "dev"], timeout=2)
            if p2p_result and "p2p" in p2p_result.lower():
                self.results["wifi"]["p2p_support"] = True

        except Exception as e:
            logger.debug(f"Linux WiFi probe failed: {e}")

    async def _probe_wifi_windows(self):
        """Windows WiFi detection"""
        try:
            # Check WiFi adapters via netsh
            result = await self._run_command_timeout(
                ["netsh", "wlan", "show", "interfaces"], timeout=5
            )

            if result:
                interfaces = []
                for line in result.split("\n"):
                    if "Name" in line and "Wi-Fi" in line:
                        interfaces.append(line.strip())

                self.results["wifi"]["interfaces"] = interfaces
                self.results["wifi"]["available"] = len(interfaces) > 0

            # Check WiFi Direct support
            wd_result = await self._run_command_timeout(
                ["netsh", "wlan", "show", "wlanreport"], timeout=3
            )

            if wd_result and "wifi direct" in wd_result.lower():
                self.results["wifi"]["wifi_direct_support"] = True

        except Exception as e:
            logger.debug(f"Windows WiFi probe failed: {e}")

    async def _probe_wifi_macos(self):
        """MacOS WiFi detection"""
        try:
            # Use networksetup to check WiFi
            result = await self._run_command_timeout(
                ["networksetup", "-listallhardwareports"], timeout=3
            )

            if result and "wi-fi" in result.lower():
                self.results["wifi"]["available"] = True
                self.results["wifi"]["details"]["hardware_ports"] = result[:300]

        except Exception as e:
            logger.debug(f"macOS WiFi probe failed: {e}")

    async def _probe_network_interfaces(self):
        """Probe general network interface capabilities"""
        try:
            interfaces = []

            if platform.system() == "Windows":
                result = await self._run_command_timeout(
                    ["ipconfig", "/all"], timeout=5
                )
                if result:
                    # Parse Windows interface information
                    current_interface = None
                    for line in result.split("\n"):
                        line = line.strip()
                        if "adapter" in line.lower() and ":" in line:
                            current_interface = line
                        elif current_interface and "Physical Address" in line:
                            interfaces.append(
                                {
                                    "name": current_interface,
                                    "mac": line.split(":")[-1].strip(),
                                }
                            )
            else:
                # Unix-like systems
                result = await self._run_command_timeout(["ip", "addr"], timeout=3)
                if not result:
                    # Fallback to ifconfig
                    result = await self._run_command_timeout(["ifconfig"], timeout=3)

                if result:
                    # Parse interface information
                    current_interface = None
                    for line in result.split("\n"):
                        if not line.startswith(" ") and ":" in line:
                            current_interface = line.split(":")[0]
                        elif current_interface and "ether" in line.lower():
                            mac = (
                                line.split()[1] if len(line.split()) > 1 else "unknown"
                            )
                            interfaces.append({"name": current_interface, "mac": mac})

            self.results["network"]["interfaces"] = interfaces

            # Determine P2P readiness
            has_wifi = self.results["wifi"]["available"]
            has_bluetooth = self.results["bluetooth"].get("available", False)
            len(interfaces) > 0

            self.results["network"]["p2p_ready"] = has_wifi or has_bluetooth
            self.results["network"]["interface_count"] = len(interfaces)

        except Exception as e:
            logger.debug(f"Network interface probe failed: {e}")
            self.results["network"]["error"] = str(e)

    async def _run_command_timeout(
        self, command: list[str], timeout: int = 5
    ) -> str | None:
        """Run command with timeout protection"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            if process.returncode == 0:
                return stdout.decode("utf-8", errors="ignore")
            logger.debug(
                f"Command failed: {' '.join(command)} - {stderr.decode()[:100]}"
            )
            return None

        except asyncio.TimeoutError:
            logger.debug(f"Command timed out: {' '.join(command)}")
            try:
                process.terminate()
                await process.wait()
            except:
                pass
            return None

        except FileNotFoundError:
            logger.debug(f"Command not found: {command[0]}")
            return None

        except Exception as e:
            logger.debug(f"Command error: {' '.join(command)} - {e}")
            return None

    def _determine_hardware_status(self):
        """Determine overall hardware status and recommendations"""
        bluetooth_ok = self.results["bluetooth"].get("available", False)
        wifi_ok = self.results["wifi"].get("available", False)
        network_ok = len(self.results["network"]["interfaces"]) > 0

        # Check for any available P2P transport
        p2p_transports = []
        recommendations = []

        if bluetooth_ok:
            p2p_transports.append("Bluetooth/BLE")
        elif self.results["bluetooth"].get("adapter_present", False):
            recommendations.append(
                "Bluetooth adapter present but may need driver/software setup"
            )
        else:
            recommendations.append(
                "No Bluetooth adapter detected - consider USB Bluetooth dongle"
            )

        if wifi_ok:
            p2p_transports.append("WiFi")
            if self.results["wifi"].get("p2p_support") or self.results["wifi"].get(
                "wifi_direct_support"
            ):
                p2p_transports.append("WiFi Direct/P2P")
        else:
            recommendations.append("WiFi adapter needed for Betanet transport")

        if network_ok:
            p2p_transports.append("Network interfaces")
        else:
            recommendations.append("No network interfaces detected")

        # Determine overall status
        hardware_ok = bluetooth_ok or wifi_ok

        self.results["summary"] = {
            "hardware_ok": hardware_ok,
            "available_transports": p2p_transports,
            "transport_count": len(p2p_transports),
            "reason": "Hardware detection complete",
            "recommendations": recommendations,
            "platform_supported": platform.system() in ["Linux", "Windows", "Darwin"],
        }

        # Add specific recommendations based on findings
        if not hardware_ok:
            if not bluetooth_ok and not wifi_ok:
                self.results["summary"]["reason"] = "No P2P capable hardware detected"
            elif not bluetooth_ok:
                self.results["summary"]["reason"] = (
                    "WiFi available but no Bluetooth for BitChat"
                )
            elif not wifi_ok:
                self.results["summary"]["reason"] = (
                    "Bluetooth available but no WiFi for Betanet"
                )


def print_hardware_status(results: dict[str, Any], verbose: bool = False):
    """Print hardware status in required format"""
    hardware_ok = results["summary"]["hardware_ok"]

    if hardware_ok:
        print("HARDWARE_OK")
        if verbose:
            transports = results["summary"]["available_transports"]
            print(f"# Available transports: {', '.join(transports)}")
            print(f"# Platform: {results['platform']['system']}")
    else:
        print("SKIPPED")
        if verbose:
            reason = results["summary"]["reason"]
            print(f"# Reason: {reason}")
            if results["summary"]["recommendations"]:
                for rec in results["summary"]["recommendations"][:2]:  # Limit output
                    print(f"# Suggestion: {rec}")


def print_detailed_report(results: dict[str, Any]):
    """Print detailed hardware report"""
    print("\n" + "=" * 60)
    print("BitChat/Betanet Hardware Probe Report")
    print("=" * 60)

    # Platform info
    platform_info = results["platform"]
    print(f"\nPlatform: {platform_info['system']} {platform_info['version']}")
    print(f"Architecture: {platform_info['architecture']}")

    # Bluetooth status
    print("\n📱 Bluetooth/BLE Status")
    bluetooth = results["bluetooth"]
    if bluetooth.get("available"):
        print("   ✅ Available and functional")
        if "discovery_test" in bluetooth:
            test = bluetooth["discovery_test"]
            if test.get("discovery_working"):
                print(
                    f"   📡 Discovery test: {test['devices_found']} devices found in {test['discovery_time_seconds']}s"
                )
            else:
                print(
                    f"   ⚠️  Discovery test failed: {test.get('error', 'Unknown error')}"
                )
    else:
        print("   ❌ Not available")
        if bluetooth.get("adapter_present"):
            print("   ℹ️  Adapter detected but not functional")
        else:
            print("   ℹ️  No adapter detected")

    # WiFi status
    print("\n📶 WiFi Status")
    wifi = results["wifi"]
    if wifi.get("available"):
        print("   ✅ Available")
        interfaces = wifi.get("interfaces", [])
        if interfaces:
            print(f"   📡 Interfaces: {len(interfaces)}")
        if wifi.get("p2p_support") or wifi.get("wifi_direct_support"):
            print("   🔗 WiFi Direct/P2P support detected")
    else:
        print("   ❌ Not available")

    # Network interfaces
    print("\n🌐 Network Interfaces")
    network = results["network"]
    interfaces = network.get("interfaces", [])
    if interfaces:
        print(f"   ✅ {len(interfaces)} interfaces detected")
        for _i, iface in enumerate(interfaces[:3]):  # Show first 3
            name = iface.get("name", "Unknown")[:30]
            print(f"   📡 {name}")
    else:
        print("   ❌ No interfaces detected")

    # Overall assessment
    print("\n🎯 Overall Assessment")
    summary = results["summary"]
    if summary["hardware_ok"]:
        print("   ✅ HARDWARE_OK - P2P networking capable")
        print(
            f"   📡 Available transports: {', '.join(summary['available_transports'])}"
        )
    else:
        print("   ❌ SKIPPED - Limited P2P capability")
        print(f"   📝 Reason: {summary['reason']}")

    # Recommendations
    if summary.get("recommendations"):
        print("\n💡 Recommendations")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")

    print("=" * 60)


async def main():
    """Main hardware probe execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hardware probe for BitChat/Betanet P2P networking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/p2p/hw_probe.py                    # Quick status check
  python tools/p2p/hw_probe.py --verbose          # Verbose output
  python tools/p2p/hw_probe.py --detailed         # Full detailed report
  python tools/p2p/hw_probe.py --json             # JSON output
  python tools/p2p/hw_probe.py --timeout 15       # Custom timeout
  python tools/p2p/hw_probe.py || true            # Safe execution (bash)
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose status output"
    )

    parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed hardware report"
    )

    parser.add_argument(
        "--json", "-j", action="store_true", help="Output results as JSON"
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=10,
        help="Probe timeout in seconds (default: 10)",
    )

    args = parser.parse_args()

    # Run hardware probe
    probe = HardwareProbe(timeout_seconds=args.timeout)

    try:
        results = await probe.probe_all_hardware()

        if args.json:
            # JSON output
            print(json.dumps(results, indent=2))
        elif args.detailed:
            # Detailed report
            print_detailed_report(results)
        else:
            # Standard status output
            print_hardware_status(results, verbose=args.verbose)

        # Exit with appropriate code
        exit_code = 0 if results["summary"]["hardware_ok"] else 1
        return exit_code

    except KeyboardInterrupt:
        print("SKIPPED")
        print("# Interrupted by user")
        return 1

    except Exception as e:
        print("SKIPPED")
        print(f"# Error: {e!s}")
        logger.error(f"Hardware probe failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception:
        print("SKIPPED")
        sys.exit(1)
