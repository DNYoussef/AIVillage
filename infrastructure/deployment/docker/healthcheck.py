#!/usr/bin/env python3

"""
Health Check Script for Python BetaNet Bridge Service
Validates BetaNet connectivity, constitutional validation, and service health
"""

import asyncio
import json
import logging
import os
import socket
import sys
import time
import traceback
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 9876))
CONSTITUTIONAL_TIER = os.getenv('CONSTITUTIONAL_TIER', 'Silver')
PRIVACY_MODE = os.getenv('PRIVACY_MODE', 'enhanced')
TIMEOUT = 10  # seconds


class HealthChecker:
    """Comprehensive health checker for Python BetaNet Bridge"""

    def __init__(self):
        self.results = {
            'healthy': True,
            'timestamp': time.time(),
            'checks': {},
            'version': '1.0.0',
            'service': 'betanet-bridge'
        }

    async def check_tcp_port(self) -> Dict[str, Any]:
        """Check if the JSON-RPC port is accessible"""
        try:
            # Create a socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)

            result = sock.connect_ex((HOST, PORT))
            sock.close()

            if result == 0:
                return {'healthy': True, 'port': PORT, 'status': 'accessible'}
            else:
                return {'healthy': False, 'port': PORT, 'status': 'not_accessible', 'error_code': result}

        except Exception as e:
            return {'healthy': False, 'port': PORT, 'error': str(e)}

    async def check_json_rpc_service(self) -> Dict[str, Any]:
        """Check if the JSON-RPC service is responding"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(HOST, PORT),
                timeout=TIMEOUT
            )

            # Send a health check request
            request = {
                'jsonrpc': '2.0',
                'id': 'health_check',
                'method': 'get_health_status',
                'params': {}
            }

            message = json.dumps(request) + '\n'
            writer.write(message.encode())
            await writer.drain()

            # Read response
            response_data = await asyncio.wait_for(
                reader.readline(),
                timeout=5
            )

            writer.close()
            await writer.wait_closed()

            if response_data:
                response = json.loads(response_data.decode())

                if 'result' in response:
                    health_data = response['result']
                    return {
                        'healthy': health_data.get('healthy', False),
                        'bridge_status': health_data.get('bridge_status'),
                        'betanet_status': health_data.get('betanet_status'),
                        'uptime': health_data.get('uptime'),
                        'metrics': health_data.get('metrics', {})
                    }
                else:
                    return {'healthy': False, 'error': 'No result in response'}
            else:
                return {'healthy': False, 'error': 'No response received'}

        except asyncio.TimeoutError:
            return {'healthy': False, 'error': 'Service timeout'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    async def check_betanet_infrastructure(self) -> Dict[str, Any]:
        """Check if BetaNet infrastructure modules are available"""
        try:
            # Try to import BetaNet modules
            sys.path.insert(0, '/app')

            from infrastructure.p2p.betanet.constitutional_transport import (
                ConstitutionalBetaNetTransport
            )
            from infrastructure.p2p.betanet.constitutional_frames import (
                ConstitutionalFrameProcessor
            )

            return {
                'healthy': True,
                'constitutional_transport': 'available',
                'frame_processor': 'available',
                'pythonpath': '/app' in sys.path
            }

        except ImportError as e:
            return {
                'healthy': False,
                'error': f'BetaNet infrastructure import failed: {str(e)}',
                'pythonpath': sys.path[:3]  # Show first 3 paths for debugging
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    async def check_constitutional_tier_config(self) -> Dict[str, Any]:
        """Validate constitutional tier configuration"""
        try:
            valid_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
            valid_privacy_modes = ['standard', 'enhanced', 'maximum']

            checks = {
                'constitutional_tier': CONSTITUTIONAL_TIER in valid_tiers,
                'privacy_mode': PRIVACY_MODE in valid_privacy_modes,
                'config': {
                    'tier': CONSTITUTIONAL_TIER,
                    'privacy': PRIVACY_MODE
                }
            }

            return {
                'healthy': all(checks.values()),
                **checks
            }

        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check Python process memory usage"""
        try:
            import psutil
            process = psutil.Process()

            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Convert to MB
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024

            # Health thresholds
            healthy = memory_percent < 80 and rss_mb < 1024  # Less than 80% and 1GB

            return {
                'healthy': healthy,
                'rss_mb': round(rss_mb, 2),
                'vms_mb': round(vms_mb, 2),
                'percent': round(memory_percent, 2),
                'warning': memory_percent > 70 or rss_mb > 512
            }

        except ImportError:
            # psutil not available, do basic check
            import resource

            memory_usage = resource.getrusage(resource.RUSAGE_SELF)
            max_rss_mb = memory_usage.ru_maxrss / 1024  # Convert to MB on Linux

            return {
                'healthy': max_rss_mb < 1024,
                'max_rss_mb': round(max_rss_mb, 2),
                'note': 'Basic memory check (psutil not available)'
            }
        except Exception as e:
            return {'healthy': True, 'error': str(e), 'note': 'Memory check failed but not critical'}

    async def check_asyncio_health(self) -> Dict[str, Any]:
        """Check asyncio event loop health"""
        try:
            loop = asyncio.get_event_loop()

            # Simple async operation timing
            start_time = time.perf_counter()
            await asyncio.sleep(0.001)  # 1ms sleep
            actual_sleep = time.perf_counter() - start_time

            # Check for significant delay (indicates loop lag)
            lag_ms = (actual_sleep - 0.001) * 1000
            healthy = lag_ms < 10  # Less than 10ms lag

            return {
                'healthy': healthy,
                'loop_lag_ms': round(lag_ms, 2),
                'loop_running': loop.is_running(),
                'warning': lag_ms > 5
            }

        except Exception as e:
            return {'healthy': True, 'error': str(e), 'note': 'Asyncio check failed but not critical'}

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and compile results"""
        try:
            logger.info("Starting BetaNet bridge health checks...")

            # Critical checks (must pass)
            logger.info("Checking TCP port accessibility...")
            self.results['checks']['tcp_port'] = await self.check_tcp_port()

            logger.info("Checking JSON-RPC service...")
            self.results['checks']['json_rpc'] = await self.check_json_rpc_service()

            logger.info("Checking BetaNet infrastructure...")
            self.results['checks']['betanet_infrastructure'] = await self.check_betanet_infrastructure()

            logger.info("Checking constitutional configuration...")
            self.results['checks']['constitutional_config'] = await self.check_constitutional_tier_config()

            # Non-critical checks (warnings only)
            logger.info("Checking memory usage...")
            self.results['checks']['memory'] = await self.check_memory_usage()

            logger.info("Checking asyncio health...")
            self.results['checks']['asyncio'] = await self.check_asyncio_health()

            # Determine overall health
            critical_checks = [
                self.results['checks']['tcp_port']['healthy'],
                self.results['checks']['json_rpc']['healthy'],
                self.results['checks']['betanet_infrastructure']['healthy'],
                self.results['checks']['constitutional_config']['healthy']
            ]

            self.results['healthy'] = all(critical_checks)

            # Count warnings
            warning_count = sum(1 for check in self.results['checks'].values()
                              if isinstance(check, dict) and check.get('warning', False))

            if warning_count > 0:
                self.results['warnings'] = warning_count

            logger.info(f"Health check completed. Healthy: {self.results['healthy']}")

            return self.results

        except Exception as e:
            logger.error(f"Health check failed with exception: {e}")
            logger.error(traceback.format_exc())

            self.results['healthy'] = False
            self.results['error'] = str(e)
            self.results['traceback'] = traceback.format_exc()

            return self.results


async def main():
    """Main health check entry point"""
    try:
        checker = HealthChecker()
        results = await checker.run_all_checks()

        # Output results
        if os.getenv('BETANET_BRIDGE_MODE') == 'production':
            # JSON output for production monitoring
            print(json.dumps(results, indent=2))
        else:
            # Human-readable output for development
            print("\n=== BetaNet Bridge Health Check ===")
            print(f"Overall Status: {'✓ HEALTHY' if results['healthy'] else '✗ UNHEALTHY'}")
            print(f"Timestamp: {time.ctime(results['timestamp'])}")

            for check_name, check_result in results['checks'].items():
                if isinstance(check_result, dict):
                    status = '✓' if check_result.get('healthy', False) else '✗'
                    warning = '⚠️' if check_result.get('warning', False) else ''
                    print(f"  {status} {check_name.replace('_', ' ').title()} {warning}")

                    if not check_result.get('healthy', False) and 'error' in check_result:
                        print(f"    Error: {check_result['error']}")

        # Exit with appropriate code
        sys.exit(0 if results['healthy'] else 1)

    except KeyboardInterrupt:
        logger.info("Health check interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to run health check: {e}")
        sys.exit(1)