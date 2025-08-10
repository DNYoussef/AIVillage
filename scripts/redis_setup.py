#!/usr/bin/env python3
"""Redis connection setup for CODEX Integration Requirements.

Configures Redis connections for all databases:
- Database 0: Evolution metrics real-time data
- Database 1: RAG pipeline caching
- Database 2: P2P peer discovery cache
"""

import logging

logger = logging.getLogger(__name__)

# Redis configuration from CODEX Integration Requirements
REDIS_CONFIGS = {
    "evolution_metrics": {
        "db": 0,
        "description": "Evolution metrics real-time data",
        "url": "redis://localhost:6379/0",
        "host": "localhost",
        "port": 6379,
        "decode_responses": True,
    },
    "rag_pipeline": {
        "db": 1,
        "description": "RAG pipeline caching",
        "url": "redis://localhost:6379/1",
        "host": "localhost",
        "port": 6379,
        "decode_responses": True,
    },
    "p2p_discovery": {
        "db": 2,
        "description": "P2P peer discovery cache",
        "url": "redis://localhost:6379/2",
        "host": "localhost",
        "port": 6379,
        "decode_responses": True,
    },
}


class RedisConnectionManager:
    """Manages Redis connections for CODEX integration."""

    def __init__(self):
        self.connections: dict[str, object | None] = {}
        self.redis_available = False
        self.redis_client = None

    def check_redis_availability(self) -> bool:
        """Check if Redis server is available."""
        try:
            import redis

            client = redis.Redis(
                host=REDIS_CONFIGS["evolution_metrics"]["host"],
                port=REDIS_CONFIGS["evolution_metrics"]["port"],
                decode_responses=True,
            )

            # Test connection
            client.ping()
            self.redis_available = True
            self.redis_client = redis
            logger.info("Redis server is available")
            return True

        except ImportError:
            logger.warning(
                "Redis package not installed. Install with: pip install redis"
            )
            return False
        except Exception as e:
            logger.warning(f"Redis server not available: {e}")
            return False

    def create_connection(self, config_name: str) -> object | None:
        """Create Redis connection for specified configuration."""
        if not self.redis_available:
            logger.warning(
                f"Cannot create Redis connection for {config_name}: Redis not available"
            )
            return None

        config = REDIS_CONFIGS[config_name]

        try:
            connection = self.redis_client.Redis(
                host=config["host"],
                port=config["port"],
                db=config["db"],
                decode_responses=config["decode_responses"],
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test the connection
            connection.ping()

            self.connections[config_name] = connection
            logger.info(
                f"Created Redis connection for {config_name} (db={config['db']})"
            )
            return connection

        except Exception as e:
            logger.error(f"Failed to create Redis connection for {config_name}: {e}")
            return None

    def setup_all_connections(self) -> dict[str, object | None]:
        """Set up all Redis connections."""
        if not self.check_redis_availability():
            logger.warning(
                "Redis not available - using SQLite fallback for all operations"
            )
            return {}

        for config_name in REDIS_CONFIGS:
            self.create_connection(config_name)

        return self.connections

    def test_connections(self) -> dict[str, bool]:
        """Test all Redis connections."""
        results = {}

        for config_name, connection in self.connections.items():
            if connection is None:
                results[config_name] = False
                continue

            try:
                connection.ping()
                results[config_name] = True
                logger.info(f"Redis connection test passed: {config_name}")
            except Exception as e:
                results[config_name] = False
                logger.error(f"Redis connection test failed for {config_name}: {e}")

        return results

    def initialize_databases(self):
        """Initialize Redis databases with initial data structures."""
        for config_name, connection in self.connections.items():
            if connection is None:
                continue

            try:
                config = REDIS_CONFIGS[config_name]

                if config_name == "evolution_metrics":
                    # Initialize evolution metrics structures
                    connection.hset(
                        "evolution:config",
                        mapping={
                            "flush_threshold": 50,
                            "current_round": 0,
                            "active_agents": 0,
                        },
                    )

                elif config_name == "rag_pipeline":
                    # Initialize RAG cache structures
                    connection.hset(
                        "rag:config",
                        mapping={
                            "cache_enabled": "true",
                            "l1_cache_size": 128,
                            "hit_rate": 0.0,
                        },
                    )

                elif config_name == "p2p_discovery":
                    # Initialize P2P discovery structures
                    connection.hset(
                        "p2p:config",
                        mapping={
                            "max_peers": 50,
                            "discovery_interval": 30,
                            "active_peers": 0,
                        },
                    )

                logger.info(f"Initialized Redis database: {config_name}")

            except Exception as e:
                logger.error(f"Failed to initialize Redis database {config_name}: {e}")

    def get_connection_info(self) -> dict:
        """Get information about Redis connections."""
        info = {
            "redis_available": self.redis_available,
            "connections": {},
            "server_info": None,
        }

        for config_name, config in REDIS_CONFIGS.items():
            connection_info = {
                "database": config["db"],
                "description": config["description"],
                "url": config["url"],
                "connected": config_name in self.connections
                and self.connections[config_name] is not None,
            }

            if connection_info["connected"]:
                try:
                    conn = self.connections[config_name]
                    connection_info["ping"] = conn.ping()
                    connection_info["db_size"] = conn.dbsize()
                except Exception as e:
                    connection_info["error"] = str(e)

            info["connections"][config_name] = connection_info

        # Get Redis server info if available
        if self.redis_available and "evolution_metrics" in self.connections:
            try:
                conn = self.connections["evolution_metrics"]
                server_info = conn.info("server")
                info["server_info"] = {
                    "version": server_info.get("redis_version"),
                    "mode": server_info.get("redis_mode"),
                    "uptime": server_info.get("uptime_in_seconds"),
                }
            except Exception as e:
                info["server_info"] = {"error": str(e)}

        return info

    def close_all_connections(self):
        """Close all Redis connections."""
        for config_name, connection in self.connections.items():
            if connection:
                try:
                    connection.close()
                    logger.info(f"Closed Redis connection: {config_name}")
                except Exception as e:
                    logger.error(f"Error closing Redis connection {config_name}: {e}")

        self.connections.clear()


def main():
    """Main Redis setup function."""
    logging.basicConfig(level=logging.INFO)

    print("Setting up Redis connections for CODEX integration...")

    manager = RedisConnectionManager()

    # Setup connections
    connections = manager.setup_all_connections()

    if not connections:
        print("\n❌ Redis not available - will use SQLite fallback")
        return False

    # Test connections
    test_results = manager.test_connections()

    # Initialize databases
    manager.initialize_databases()

    # Get connection info
    info = manager.get_connection_info()

    # Print summary
    print("\n" + "=" * 60)
    print("REDIS CONNECTION SETUP COMPLETE")
    print("=" * 60)

    if info["server_info"] and "version" in info["server_info"]:
        server_info = info["server_info"]
        print("\nRedis Server Info:")
        print(f"  Version: {server_info.get('version', 'Unknown')}")
        print(f"  Mode: {server_info.get('mode', 'Unknown')}")
        print(f"  Uptime: {server_info.get('uptime', 0)} seconds")

    print("\nDatabase Connections:")
    for config_name, conn_info in info["connections"].items():
        status = "✅ CONNECTED" if conn_info["connected"] else "❌ FAILED"
        print(f"  {config_name} (DB {conn_info['database']}): {status}")
        print(f"    Description: {conn_info['description']}")
        if conn_info["connected"] and "db_size" in conn_info:
            print(f"    Records: {conn_info['db_size']}")
        if "error" in conn_info:
            print(f"    Error: {conn_info['error']}")

    print("\nTest Results:")
    passed = sum(test_results.values())
    total = len(test_results)
    print(f"  Passed: {passed}/{total} connections")

    print("\n" + "=" * 60)
    if passed == total:
        print("Redis setup completed successfully!")
    else:
        print("Redis setup completed with some failures - check logs")
    print("=" * 60)

    # Close connections
    manager.close_all_connections()

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
