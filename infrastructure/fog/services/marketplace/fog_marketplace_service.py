"""
Fog Marketplace Service

Manages the fog computing service marketplace including:
- Service registration and discovery
- Request handling and contract management
- Dynamic pricing and spot pricing
- SLA tier management and compliance
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from decimal import Decimal

from ..interfaces.base_service import BaseFogService, ServiceStatus, ServiceHealthCheck
from ...marketplace.fog_marketplace import FogMarketplace, ServiceOffering, ServiceRequest, ServiceType, ServiceTier
from ...scheduler.enhanced_sla_tiers import EnhancedSLATierManager, SLAMetrics, SLATier


class FogMarketplaceService(BaseFogService):
    """Service for managing fog computing marketplace"""
    
    def __init__(self, service_name: str, config: Dict[str, Any], event_bus):
        super().__init__(service_name, config, event_bus)
        
        # Core components
        self.marketplace: Optional[FogMarketplace] = None
        self.sla_tier_manager: Optional[EnhancedSLATierManager] = None
        
        # Marketplace configuration
        self.marketplace_config = config.get("marketplace", {})
        self.node_id = config.get("node_id", "default")
        
        # Service metrics
        self.metrics = {
            "total_offerings": 0,
            "active_contracts": 0,
            "completed_contracts": 0,
            "failed_contracts": 0,
            "total_revenue": 0.0,
            "spot_price_updates": 0,
            "sla_violations": 0,
            "services_by_tier": {
                "bronze": 0,
                "silver": 0, 
                "gold": 0
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the marketplace service"""
        try:
            # Initialize fog marketplace
            self.marketplace = FogMarketplace(
                marketplace_id=f"fog-market-{self.node_id}",
                base_token_rate=self.marketplace_config.get("base_token_rate", 100),
                enable_hidden_services=self.marketplace_config.get("enable_hidden_services", True),
                enable_spot_pricing=self.marketplace_config.get("enable_spot_pricing", True),
            )
            
            # Initialize SLA tier manager (will be injected as dependency)
            # self.sla_tier_manager will be set via dependency injection
            
            # Subscribe to relevant events
            self.subscribe_to_events("service_offering", self._handle_service_offering)
            self.subscribe_to_events("service_request", self._handle_service_request)
            self.subscribe_to_events("sla_violation", self._handle_sla_violation)
            self.subscribe_to_events("contract_completed", self._handle_contract_completed)
            
            # Start background tasks
            self.add_background_task(self._pricing_update_task(), "pricing_updates")
            self.add_background_task(self._sla_monitoring_task(), "sla_monitoring")
            self.add_background_task(self._marketplace_analytics_task(), "analytics")
            
            self.logger.info("Fog marketplace service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize marketplace service: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup marketplace service resources"""
        try:
            # Cancel any active contracts gracefully
            if self.marketplace:
                # Implementation would handle contract cleanup
                pass
            
            self.logger.info("Fog marketplace service cleaned up")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up marketplace service: {e}")
            return False
    
    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check on marketplace service"""
        try:
            error_messages = []
            
            # Check marketplace
            if not self.marketplace:
                error_messages.append("Marketplace not initialized")
            
            # Check SLA violation rate
            total_contracts = self.metrics["completed_contracts"] + self.metrics["failed_contracts"]
            if total_contracts > 0:
                violation_rate = self.metrics["sla_violations"] / total_contracts
                if violation_rate > 0.05:  # More than 5% SLA violation rate
                    error_messages.append(f"High SLA violation rate: {violation_rate:.2%}")
            
            # Check contract failure rate
            if total_contracts > 0:
                failure_rate = self.metrics["failed_contracts"] / total_contracts
                if failure_rate > 0.1:  # More than 10% failure rate
                    error_messages.append(f"High contract failure rate: {failure_rate:.2%}")
            
            status = ServiceStatus.RUNNING if not error_messages else ServiceStatus.ERROR
            
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=status,
                last_check=datetime.now(UTC),
                error_message="; ".join(error_messages) if error_messages else None,
                metrics=self.metrics.copy()
            )
            
        except Exception as e:
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.now(UTC),
                error_message=f"Health check failed: {e}",
                metrics=self.metrics.copy()
            )
    
    async def register_service_offering(
        self, 
        provider_id: str, 
        offering_data: Dict[str, Any]
    ) -> bool:
        """Register a new service offering in the marketplace"""
        try:
            if not self.marketplace:
                return False
            
            # Create service offering
            offering = ServiceOffering(
                offering_id=offering_data["offering_id"],
                provider_id=provider_id,
                service_type=ServiceType[offering_data["service_type"]],
                service_tier=ServiceTier[offering_data.get("service_tier", "BASIC")],
                pricing_model=offering_data.get("pricing_model", "HOURLY"),
                base_price=Decimal(str(offering_data.get("base_price", "0.01"))),
                regions=offering_data.get("regions", ["fog_network"]),
                uptime_guarantee=offering_data.get("uptime_guarantee", 99.0),
            )
            
            success = await self.marketplace.register_offering(provider_id, offering)
            
            if success:
                self.metrics["total_offerings"] += 1
                
                # Update tier metrics
                tier_name = offering.service_tier.value.lower()
                if tier_name in self.metrics["services_by_tier"]:
                    self.metrics["services_by_tier"][tier_name] += 1
                
                # Publish offering registration event
                await self.publish_event("offering_registered", {
                    "offering_id": offering.offering_id,
                    "provider_id": provider_id,
                    "service_type": offering.service_type.value,
                    "service_tier": offering.service_tier.value,
                    "timestamp": datetime.now(UTC).isoformat()
                })
                
                self.logger.info(f"Registered service offering: {offering.offering_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to register service offering: {e}")
            return False
    
    async def submit_service_request(
        self, 
        customer_id: str, 
        request_data: Dict[str, Any]
    ) -> Optional[str]:
        """Submit a service request to the marketplace"""
        try:
            if not self.marketplace:
                return None
            
            # Create service request
            service_request = ServiceRequest(
                request_id=request_data["request_id"],
                customer_id=customer_id,
                service_type=ServiceType[request_data["service_type"]],
                service_tier=ServiceTier[request_data.get("service_tier", "BASIC")],
            )
            
            contract_id = await self.marketplace.submit_request(customer_id, service_request)
            
            if contract_id:
                self.metrics["active_contracts"] += 1
                
                # Publish service request event
                await self.publish_event("service_request_submitted", {
                    "request_id": service_request.request_id,
                    "contract_id": contract_id,
                    "customer_id": customer_id,
                    "service_type": service_request.service_type.value,
                    "timestamp": datetime.now(UTC).isoformat()
                })
            
            return contract_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit service request: {e}")
            return None
    
    async def provision_sla_service(
        self, 
        service_id: str, 
        tier: str, 
        available_devices: List[Dict[str, Any]],
        service_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provision a service with specific SLA requirements"""
        try:
            if not self.sla_tier_manager:
                return {"success": False, "error": "SLA tier manager not available"}
            
            sla_tier = SLATier[tier.upper()]
            
            result = await self.sla_tier_manager.provision_service(
                service_id=service_id,
                tier=sla_tier,
                available_devices=available_devices,
                service_config=service_config,
            )
            
            if result.get("success"):
                # Update tier metrics
                tier_name = tier.lower()
                if tier_name in self.metrics["services_by_tier"]:
                    self.metrics["services_by_tier"][tier_name] += 1
                
                # Publish SLA service provision event
                await self.publish_event("sla_service_provisioned", {
                    "service_id": service_id,
                    "tier": tier,
                    "timestamp": datetime.now(UTC).isoformat()
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to provision SLA service: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_sla_compliance(
        self, 
        service_id: str, 
        metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate SLA compliance for a service"""
        try:
            if not self.sla_tier_manager:
                return {"compliant": False, "error": "SLA tier manager not available"}
            
            metrics = SLAMetrics(**metrics_data)
            result = await self.sla_tier_manager.validate_sla_compliance(service_id, metrics)
            
            if not result.get("compliant"):
                self.metrics["sla_violations"] += 1
                
                # Publish SLA violation event
                await self.publish_event("sla_violation_detected", {
                    "service_id": service_id,
                    "violations": result.get("violations", []),
                    "timestamp": datetime.now(UTC).isoformat()
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to validate SLA compliance: {e}")
            return {"compliant": False, "error": str(e)}
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get comprehensive marketplace statistics"""
        try:
            stats = self.metrics.copy()
            
            if self.marketplace:
                market_stats = self.marketplace.get_market_stats()
                stats.update({
                    "marketplace_stats": market_stats.__dict__,
                    "total_providers": len(getattr(self.marketplace, 'providers', {})),
                    "total_customers": len(getattr(self.marketplace, 'customers', {}))
                })
            
            if self.sla_tier_manager:
                sla_status = self.sla_tier_manager.get_all_services_status()
                stats["sla_services"] = sla_status
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get marketplace stats: {e}")
            return self.metrics.copy()
    
    async def _handle_service_offering(self, event):
        """Handle service offering registration events"""
        provider_id = event.data.get("provider_id")
        offering_data = event.data.get("offering_data", {})
        
        success = await self.register_service_offering(provider_id, offering_data)
        
        await self.publish_event("service_offering_response", {
            "request_id": event.data.get("request_id"),
            "success": success
        })
    
    async def _handle_service_request(self, event):
        """Handle service request submissions"""
        customer_id = event.data.get("customer_id")
        request_data = event.data.get("request_data", {})
        
        contract_id = await self.submit_service_request(customer_id, request_data)
        
        await self.publish_event("service_request_response", {
            "request_id": event.data.get("request_id"),
            "contract_id": contract_id,
            "success": contract_id is not None
        })
    
    async def _handle_sla_violation(self, event):
        """Handle SLA violation notifications"""
        service_id = event.data.get("service_id")
        violations = event.data.get("violations", [])
        
        self.logger.warning(f"SLA violation for service {service_id}: {violations}")
        
        # Implement remediation logic here
        # Could trigger service rebalancing, provider penalties, etc.
    
    async def _handle_contract_completed(self, event):
        """Handle contract completion events"""
        contract_id = event.data.get("contract_id")
        success = event.data.get("success", False)
        revenue = event.data.get("revenue", 0.0)
        
        self.metrics["active_contracts"] -= 1
        
        if success:
            self.metrics["completed_contracts"] += 1
            self.metrics["total_revenue"] += revenue
        else:
            self.metrics["failed_contracts"] += 1
        
        self.logger.info(f"Contract {contract_id} completed: {'success' if success else 'failed'}")
    
    async def _pricing_update_task(self):
        """Background task to update marketplace pricing"""
        while not self._shutdown_event.is_set():
            try:
                if self.marketplace:
                    # Update dynamic pricing for all service types
                    for service_type in ServiceType:
                        await self.marketplace._update_dynamic_pricing(service_type)
                        self.metrics["spot_price_updates"] += 1
                
                await asyncio.sleep(900)  # Update every 15 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Pricing update error: {e}")
                await asyncio.sleep(300)
    
    async def _sla_monitoring_task(self):
        """Background task to monitor SLA compliance"""
        while not self._shutdown_event.is_set():
            try:
                if self.sla_tier_manager:
                    # Get all active services
                    all_services = self.sla_tier_manager.get_all_services_status()
                    
                    # Check compliance for each service
                    for tier_name, services in all_services.get("services_by_tier", {}).items():
                        for service in services:
                            service_id = service["service_id"]
                            
                            # Mock metrics (in production, collect from monitoring)
                            current_metrics = SLAMetrics(
                                p95_latency_ms=100.0,
                                uptime_percentage=99.5,
                                error_rate_percentage=0.05,
                                throughput_ops_per_second=1000.0,
                            )
                            
                            # Validate compliance
                            compliance_result = await self.validate_sla_compliance(
                                service_id, current_metrics.__dict__
                            )
                            
                            if not compliance_result.get("compliant"):
                                # Handle SLA violations
                                await self._handle_sla_violation_remediation(
                                    service_id, compliance_result
                                )
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"SLA monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _marketplace_analytics_task(self):
        """Background task to collect marketplace analytics"""
        while not self._shutdown_event.is_set():
            try:
                # Update analytics metrics
                if self.marketplace:
                    market_stats = self.marketplace.get_market_stats()
                    self.metrics["total_offerings"] = market_stats.total_offerings
                    
                    # Publish analytics update
                    await self.publish_event("marketplace_analytics", {
                        "metrics": self.metrics.copy(),
                        "timestamp": datetime.now(UTC).isoformat()
                    })
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Analytics task error: {e}")
                await asyncio.sleep(120)
    
    async def _handle_sla_violation_remediation(
        self, 
        service_id: str, 
        compliance_result: Dict[str, Any]
    ):
        """Handle SLA violation remediation"""
        try:
            violations = compliance_result.get("violations", [])
            
            # Implement remediation strategies
            for violation in violations:
                if "diversity" in violation.lower() and self.sla_tier_manager:
                    # Attempt rebalancing for diversity violations
                    available_devices = []  # Would get from device registry
                    if available_devices:
                        rebalance_result = await self.sla_tier_manager.rebalance_service(
                            service_id, available_devices
                        )
                        if rebalance_result.get("success"):
                            self.logger.info(f"Successfully rebalanced service {service_id}")
                
                # Other remediation strategies could be implemented here
                
        except Exception as e:
            self.logger.error(f"Error in SLA violation remediation: {e}")