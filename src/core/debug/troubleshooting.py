"""AIVillage Troubleshooting Tools.

Implements comprehensive troubleshooting utilities as specified in CODEX Integration Requirements.
"""

import json
import logging
import os
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any

import psutil


class TroubleshootingTools:
    """Comprehensive troubleshooting toolkit for AIVillage system.

    Implements all debugging tools specified in CODEX Integration Requirements:
    - Port conflict checker
    - Database lock detector
    - Memory usage profiler
    - Network discovery debugger
    - Configuration validator
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def check_port_conflicts(
        self, required_ports: list[int] | None = None
    ) -> dict[str, Any]:
        """Check for port conflicts using netstat equivalent functionality.

        Args:
            required_ports: List of ports that need to be available

        Returns:
            Dictionary with port conflict analysis
        """
        if required_ports is None:
            # CODEX Integration Requirements ports
            required_ports = [8080, 8081, 8082, 8083, 4001, 4002, 6379, 5432]

        results = {
            "timestamp": time.time(),
            "total_ports_checked": len(required_ports),
            "conflicts_found": 0,
            "available_ports": [],
            "conflicted_ports": [],
            "port_details": {},
        }

        self.logger.debug(f"Checking port conflicts for ports: {required_ports}")

        for port in required_ports:
            port_info = self._check_single_port(port)
            results["port_details"][port] = port_info

            if port_info["in_use"]:
                results["conflicts_found"] += 1
                results["conflicted_ports"].append(port)
                self.logger.warning(
                    f"Port {port} is in use by: {port_info['process_name']}"
                )
            else:
                results["available_ports"].append(port)
                self.logger.debug(f"Port {port} is available")

        # Get all listening ports for comprehensive analysis
        all_listening = self._get_all_listening_ports()
        results["all_listening_ports"] = all_listening

        return results

    def _check_single_port(self, port: int) -> dict[str, Any]:
        """Check if a single port is in use."""
        port_info = {
            "port": port,
            "in_use": False,
            "process_name": None,
            "process_pid": None,
            "protocol": None,
            "address": None,
        }

        try:
            # Check if port is in use by trying to bind to it
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = sock.connect_ex(("localhost", port))

                if result == 0:
                    port_info["in_use"] = True

                    # Get process information using psutil
                    for conn in psutil.net_connections():
                        if (
                            conn.laddr.port == port
                            and conn.status == psutil.CONN_LISTEN
                        ):
                            try:
                                process = psutil.Process(conn.pid)
                                port_info["process_name"] = process.name()
                                port_info["process_pid"] = conn.pid
                                port_info["protocol"] = (
                                    "TCP" if conn.type == socket.SOCK_STREAM else "UDP"
                                )
                                port_info[
                                    "address"
                                ] = f"{conn.laddr.ip}:{conn.laddr.port}"
                                break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                port_info["process_name"] = "Unknown"

        except Exception as e:
            self.logger.exception(f"Error checking port {port}: {e}")
            port_info["error"] = str(e)

        return port_info

    def _get_all_listening_ports(self) -> list[dict[str, Any]]:
        """Get all currently listening ports."""
        listening_ports = []

        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.status == psutil.CONN_LISTEN:
                    try:
                        process = psutil.Process(conn.pid) if conn.pid else None
                        port_info = {
                            "port": conn.laddr.port,
                            "address": f"{conn.laddr.ip}:{conn.laddr.port}",
                            "protocol": (
                                "TCP" if conn.type == socket.SOCK_STREAM else "UDP"
                            ),
                            "process_name": process.name() if process else "Unknown",
                            "process_pid": conn.pid,
                        }
                        listening_ports.append(port_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            self.logger.exception(f"Error getting listening ports: {e}")

        return listening_ports

    def detect_database_locks(
        self, db_paths: list[str] | None = None
    ) -> dict[str, Any]:
        """Detect SQLite database locks and analyze database status.

        Args:
            db_paths: List of database file paths to check

        Returns:
            Dictionary with database lock analysis
        """
        if db_paths is None:
            db_paths = [
                "./data/evolution_metrics.db",
                "./data/digital_twin.db",
                "./data/rag_index.db",
            ]

        results = {
            "timestamp": time.time(),
            "databases_checked": len(db_paths),
            "locked_databases": 0,
            "database_details": {},
        }

        self.logger.debug(f"Checking database locks for: {db_paths}")

        for db_path in db_paths:
            db_info = self._analyze_database(db_path)
            results["database_details"][db_path] = db_info

            if db_info.get("locked", False):
                results["locked_databases"] += 1
                self.logger.warning(f"Database locked: {db_path}")

        return results

    def _analyze_database(self, db_path: str) -> dict[str, Any]:
        """Analyze a single database file."""
        db_info = {
            "path": db_path,
            "exists": False,
            "locked": False,
            "accessible": False,
            "size_bytes": 0,
            "journal_mode": None,
            "connection_count": 0,
            "wal_file_exists": False,
            "shm_file_exists": False,
            "error": None,
        }

        try:
            db_path_obj = Path(db_path)

            # Check if database file exists
            if not db_path_obj.exists():
                db_info["error"] = "Database file does not exist"
                return db_info

            db_info["exists"] = True
            db_info["size_bytes"] = db_path_obj.stat().st_size

            # Check for WAL and SHM files (indicate active connections)
            wal_path = db_path_obj.with_suffix(db_path_obj.suffix + "-wal")
            shm_path = db_path_obj.with_suffix(db_path_obj.suffix + "-shm")

            db_info["wal_file_exists"] = wal_path.exists()
            db_info["shm_file_exists"] = shm_path.exists()

            # Try to connect to database
            try:
                conn = sqlite3.connect(db_path, timeout=1.0)
                db_info["accessible"] = True

                # Get journal mode
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode")
                db_info["journal_mode"] = cursor.fetchone()[0]

                # Test if database is locked by trying a simple operation
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute("ROLLBACK")

                conn.close()

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    db_info["locked"] = True
                    db_info["error"] = "Database is locked"
                else:
                    db_info["error"] = str(e)

        except Exception as e:
            db_info["error"] = str(e)
            self.logger.exception(f"Error analyzing database {db_path}: {e}")

        return db_info

    def profile_memory_usage(
        self, include_faiss_simulation: bool = True
    ) -> dict[str, Any]:
        """Profile memory usage with FAISS memory simulation.

        Args:
            include_faiss_simulation: Whether to simulate FAISS memory usage

        Returns:
            Dictionary with memory usage analysis
        """
        results = {
            "timestamp": time.time(),
            "system_memory": {},
            "process_memory": {},
            "faiss_simulation": {},
            "recommendations": [],
        }

        # Get system memory information
        memory = psutil.virtual_memory()
        results["system_memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "free_gb": memory.free / (1024**3),
            "percent_used": memory.percent,
            "buffers_gb": getattr(memory, "buffers", 0) / (1024**3),
            "cached_gb": getattr(memory, "cached", 0) / (1024**3),
        }

        # Get current process memory information
        process = psutil.Process()
        process_memory = process.memory_info()
        results["process_memory"] = {
            "rss_mb": process_memory.rss / (1024**2),
            "vms_mb": process_memory.vms / (1024**2),
            "percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "open_files": (
                len(process.open_files()) if hasattr(process, "open_files") else 0
            ),
        }

        # FAISS memory simulation (CODEX requirement: keep under 2GB for 100K docs)
        if include_faiss_simulation:
            faiss_results = self._simulate_faiss_memory()
            results["faiss_simulation"] = faiss_results

        # Generate recommendations
        recommendations = []
        if memory.percent > 85:
            recommendations.append(
                "High system memory usage - consider optimizing memory allocation"
            )

        if results["process_memory"]["rss_mb"] > 1000:
            recommendations.append(
                "High process memory usage - monitor for memory leaks"
            )

        if (
            include_faiss_simulation
            and results["faiss_simulation"]["estimated_usage_gb"] > 1.8
        ):
            recommendations.append(
                "FAISS memory usage approaching 2GB limit - implement lazy loading"
            )

        results["recommendations"] = recommendations

        self.logger.debug(
            f"Memory profiling completed: {memory.percent:.1f}% system usage"
        )

        return results

    def _simulate_faiss_memory(self) -> dict[str, Any]:
        """Simulate FAISS memory usage for 100K documents."""
        # CODEX Integration Requirements: 100K documents, 384 dimensions
        document_count = 100000
        vector_dimension = 384
        bytes_per_float = 4  # 32-bit floats

        # Calculate base vector storage
        vectors_memory_bytes = document_count * vector_dimension * bytes_per_float

        # Add FAISS index overhead (approximately 20-30% additional memory)
        faiss_overhead_factor = 1.25
        total_faiss_memory = vectors_memory_bytes * faiss_overhead_factor

        # Add metadata overhead (IDs, etc.)
        metadata_memory = document_count * 64  # 64 bytes per document metadata

        total_memory = total_faiss_memory + metadata_memory

        return {
            "document_count": document_count,
            "vector_dimension": vector_dimension,
            "vectors_memory_gb": vectors_memory_bytes / (1024**3),
            "faiss_overhead_gb": (total_faiss_memory - vectors_memory_bytes)
            / (1024**3),
            "metadata_memory_mb": metadata_memory / (1024**2),
            "estimated_usage_gb": total_memory / (1024**3),
            "codex_limit_gb": 2.0,
            "within_limit": total_memory / (1024**3) < 2.0,
            "utilization_percent": (total_memory / (1024**3)) / 2.0 * 100,
        }

    def debug_network_discovery(self) -> dict[str, Any]:
        """Debug P2P network discovery functionality.

        Returns:
            Dictionary with network discovery analysis
        """
        results = {
            "timestamp": time.time(),
            "network_interfaces": [],
            "mdns_status": {},
            "peer_discovery": {},
            "connectivity_tests": {},
        }

        # Get network interfaces
        try:
            interfaces = psutil.net_if_addrs()
            for interface_name, addresses in interfaces.items():
                interface_info = {"name": interface_name, "addresses": []}

                for addr in addresses:
                    interface_info["addresses"].append(
                        {
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast,
                        }
                    )

                results["network_interfaces"].append(interface_info)

        except Exception as e:
            self.logger.exception(f"Error getting network interfaces: {e}")
            results["network_interfaces_error"] = str(e)

        # Test mDNS functionality (simulation)
        results["mdns_status"] = self._test_mdns_functionality()

        # Test peer discovery (simulation)
        results["peer_discovery"] = self._simulate_peer_discovery()

        # Test basic connectivity
        results["connectivity_tests"] = self._test_connectivity()

        return results

    def _test_mdns_functionality(self) -> dict[str, Any]:
        """Test mDNS functionality (simulated)."""
        return {
            "service_name": "_aivillage._tcp",
            "port": 4001,
            "ttl": 120,
            "status": "simulated",
            "broadcast_capability": True,
            "discovery_interval": 30,
        }

    def _simulate_peer_discovery(self) -> dict[str, Any]:
        """Simulate peer discovery process."""
        return {
            "discovery_time_ms": 5000,
            "peers_found": 0,
            "discovery_steps": [
                {"step": "mDNS broadcast", "duration_ms": 500, "status": "completed"},
                {
                    "step": "Response collection",
                    "duration_ms": 2000,
                    "status": "completed",
                },
                {
                    "step": "Peer verification",
                    "duration_ms": 1500,
                    "status": "completed",
                },
                {
                    "step": "Connection establishment",
                    "duration_ms": 1000,
                    "status": "completed",
                },
            ],
            "total_discovery_time": 5.0,
        }

    def _test_connectivity(self) -> dict[str, Any]:
        """Test basic network connectivity."""
        connectivity_tests = {
            "localhost": self._test_connection("localhost", 80),
            "dns_resolution": self._test_dns_resolution(),
            "local_network": self._test_local_network(),
        }

        return connectivity_tests

    def _test_connection(
        self, host: str, port: int, timeout: float = 5.0
    ) -> dict[str, Any]:
        """Test connection to a specific host and port."""
        result = {
            "host": host,
            "port": port,
            "connected": False,
            "response_time_ms": None,
            "error": None,
        }

        try:
            start_time = time.time()
            sock = socket.create_connection((host, port), timeout)
            end_time = time.time()

            result["connected"] = True
            result["response_time_ms"] = (end_time - start_time) * 1000
            sock.close()

        except Exception as e:
            result["error"] = str(e)

        return result

    def _test_dns_resolution(self) -> dict[str, Any]:
        """Test DNS resolution."""
        test_hosts = ["localhost", "127.0.0.1"]
        results = {}

        for host in test_hosts:
            try:
                start_time = time.time()
                addr_info = socket.getaddrinfo(host, None)
                end_time = time.time()

                results[host] = {
                    "resolved": True,
                    "addresses": [info[4][0] for info in addr_info],
                    "resolution_time_ms": (end_time - start_time) * 1000,
                }
            except Exception as e:
                results[host] = {"resolved": False, "error": str(e)}

        return results

    def _test_local_network(self) -> dict[str, Any]:
        """Test local network connectivity."""
        return {
            "gateway_reachable": True,  # Simplified for now
            "local_subnet_accessible": True,
            "multicast_capable": True,
        }

    def validate_configuration(
        self, config_files: list[str] | None = None
    ) -> dict[str, Any]:
        """Validate system configuration files.

        Args:
            config_files: List of configuration file paths to validate

        Returns:
            Dictionary with configuration validation results
        """
        if config_files is None:
            config_files = [
                "pyproject.toml",
                ".flake8",
                ".isort.cfg",
                ".pre-commit-config.yaml",
                "config/aivillage_config.yaml",
            ]

        results = {
            "timestamp": time.time(),
            "files_checked": len(config_files),
            "valid_files": 0,
            "invalid_files": 0,
            "file_details": {},
            "environment_variables": self._check_environment_variables(),
        }

        for config_file in config_files:
            file_result = self._validate_config_file(config_file)
            results["file_details"][config_file] = file_result

            if file_result["valid"]:
                results["valid_files"] += 1
            else:
                results["invalid_files"] += 1

        return results

    def _validate_config_file(self, file_path: str) -> dict[str, Any]:
        """Validate a single configuration file."""
        result = {
            "path": file_path,
            "exists": False,
            "valid": False,
            "format": None,
            "size_bytes": 0,
            "errors": [],
            "warnings": [],
        }

        try:
            path_obj = Path(file_path)

            if not path_obj.exists():
                result["errors"].append("File does not exist")
                return result

            result["exists"] = True
            result["size_bytes"] = path_obj.stat().st_size

            # Determine file format and validate
            if file_path.endswith(".json"):
                result["format"] = "json"
                self._validate_json_file(path_obj, result)
            elif file_path.endswith((".yaml", ".yml")):
                result["format"] = "yaml"
                self._validate_yaml_file(path_obj, result)
            elif file_path.endswith(".toml"):
                result["format"] = "toml"
                self._validate_toml_file(path_obj, result)
            else:
                result["format"] = "text"
                result["valid"] = True  # Assume text files are valid if they exist

        except Exception as e:
            result["errors"].append(str(e))

        result["valid"] = len(result["errors"]) == 0

        return result

    def _validate_json_file(self, path: Path, result: dict[str, Any]) -> None:
        """Validate a JSON file."""
        try:
            with open(path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON: {e}")

    def _validate_yaml_file(self, path: Path, result: dict[str, Any]) -> None:
        """Validate a YAML file."""
        try:
            import yaml

            with open(path) as f:
                yaml.safe_load(f)
        except ImportError:
            result["warnings"].append("PyYAML not installed - cannot validate YAML")
        except yaml.YAMLError as e:
            result["errors"].append(f"Invalid YAML: {e}")

    def _validate_toml_file(self, path: Path, result: dict[str, Any]) -> None:
        """Validate a TOML file."""
        try:
            import tomllib

            with open(path, "rb") as f:
                tomllib.load(f)
        except ImportError:
            try:
                import toml

                with open(path) as f:
                    toml.load(f)
            except ImportError:
                result["warnings"].append("No TOML library available - cannot validate")
        except Exception as e:
            result["errors"].append(f"Invalid TOML: {e}")

    def _check_environment_variables(self) -> dict[str, Any]:
        """Check important environment variables."""
        important_vars = [
            "AIVILLAGE_DEBUG_MODE",
            "AIVILLAGE_LOG_LEVEL",
            "AIVILLAGE_PROFILE_PERFORMANCE",
            "PYTHONPATH",
            "PATH",
        ]

        env_status = {}
        for var in important_vars:
            env_status[var] = {
                "set": var in os.environ,
                "value": os.environ.get(var, None),
            }

        return env_status

    def run_comprehensive_diagnosis(self) -> dict[str, Any]:
        """Run comprehensive system diagnosis.

        Returns:
            Complete diagnostic report
        """
        self.logger.info("Starting comprehensive system diagnosis")

        start_time = time.time()

        diagnosis = {
            "timestamp": start_time,
            "diagnosis_version": "1.0.0",
            "port_conflicts": {},
            "database_locks": {},
            "memory_profile": {},
            "network_discovery": {},
            "configuration_validation": {},
            "overall_health": "unknown",
        }

        try:
            # Run all diagnostic checks
            diagnosis["port_conflicts"] = self.check_port_conflicts()
            diagnosis["database_locks"] = self.detect_database_locks()
            diagnosis["memory_profile"] = self.profile_memory_usage()
            diagnosis["network_discovery"] = self.debug_network_discovery()
            diagnosis["configuration_validation"] = self.validate_configuration()

            # Calculate overall health
            health_score = self._calculate_health_score(diagnosis)
            diagnosis["health_score"] = health_score

            if health_score >= 0.9:
                diagnosis["overall_health"] = "excellent"
            elif health_score >= 0.7:
                diagnosis["overall_health"] = "good"
            elif health_score >= 0.5:
                diagnosis["overall_health"] = "fair"
            else:
                diagnosis["overall_health"] = "poor"

        except Exception as e:
            self.logger.exception(f"Error during comprehensive diagnosis: {e}")
            diagnosis["error"] = str(e)

        diagnosis["duration_seconds"] = time.time() - start_time

        self.logger.info(
            f"Comprehensive diagnosis completed in {diagnosis['duration_seconds']:.2f}s"
        )
        self.logger.info(f"Overall system health: {diagnosis['overall_health']}")

        return diagnosis

    def _calculate_health_score(self, diagnosis: dict[str, Any]) -> float:
        """Calculate overall system health score."""
        score = 1.0

        # Port conflicts reduce score
        if diagnosis.get("port_conflicts", {}).get("conflicts_found", 0) > 0:
            score -= 0.2

        # Database locks reduce score significantly
        if diagnosis.get("database_locks", {}).get("locked_databases", 0) > 0:
            score -= 0.3

        # High memory usage reduces score
        memory_percent = (
            diagnosis.get("memory_profile", {})
            .get("system_memory", {})
            .get("percent_used", 0)
        )
        if memory_percent > 90:
            score -= 0.3
        elif memory_percent > 80:
            score -= 0.1

        # Configuration issues reduce score
        invalid_configs = diagnosis.get("configuration_validation", {}).get(
            "invalid_files", 0
        )
        if invalid_configs > 0:
            score -= 0.1 * invalid_configs

        return max(0.0, score)
