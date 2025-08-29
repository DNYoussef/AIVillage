"""
Emergency Triage API Endpoints

Archaeological Integration: Add triage endpoints to enhanced unified API gateway.
These endpoints integrate the Emergency Triage System with the existing API infrastructure.

Archaeological Integration Status: ACTIVE
Innovation Score: 8.0/10 (CRITICAL) 
Implementation Date: 2025-08-29
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIResponse, Depends, HTTPException
from pydantic import BaseModel, Field

from infrastructure.gateway.auth import JWTBearer, TokenPayload
from infrastructure.monitoring.triage.emergency_triage_system import (
    EmergencyTriageSystem,
    ThreatLevel,
    TriageStatus
)


# API Models for Triage System
class TriageIncidentRequest(BaseModel):
    """Request to report a new triage incident."""
    source_component: str = Field(..., description="Component where incident occurred")
    incident_type: str = Field(..., description="Type of incident")
    description: str = Field(..., description="Incident description")
    threat_level: Optional[str] = Field(default=None, description="Override threat level")
    raw_data: dict = Field(default_factory=dict, description="Additional incident data")


class TriageStatusRequest(BaseModel):
    """Request to update triage incident status."""
    incident_id: str = Field(..., description="Incident ID to update")
    new_status: str = Field(..., description="New status for incident")
    notes: Optional[str] = Field(default=None, description="Additional notes")


def register_triage_endpoints(app, service_manager, jwt_auth: JWTBearer):
    """Register Emergency Triage System endpoints with FastAPI app."""
    
    @app.post("/v1/monitoring/triage/incident", response_model=APIResponse)
    async def report_triage_incident(
        request: TriageIncidentRequest,
        token: TokenPayload = Depends(jwt_auth)
    ):
        """Report a new emergency triage incident."""
        
        if service_manager.services["emergency_triage"]["status"] != "running":
            raise HTTPException(status_code=503, detail="Emergency triage service unavailable")
        
        try:
            # Parse threat level if provided
            threat_level = None
            if request.threat_level:
                try:
                    threat_level = ThreatLevel(request.threat_level.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid threat level: {request.threat_level}")
            
            # Create triage incident
            incident = service_manager.emergency_triage.detect_incident(
                source_component=request.source_component,
                incident_type=request.incident_type,
                description=request.description,
                raw_data=request.raw_data,
                threat_level=threat_level
            )
            
            return APIResponse(
                data={
                    "incident_id": incident.incident_id,
                    "threat_level": incident.threat_level.value,
                    "status": incident.status.value,
                    "confidence_score": incident.confidence_score,
                    "timestamp": incident.timestamp.isoformat()
                },
                message=f"Triage incident {incident.incident_id} created and classified as {incident.threat_level.value}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create triage incident: {e}")
            raise HTTPException(status_code=500, detail=f"Triage incident creation failed: {str(e)}")
    
    
    @app.get("/v1/monitoring/triage/incidents", response_model=APIResponse)
    async def get_triage_incidents(
        status: Optional[str] = None,
        threat_level: Optional[str] = None,
        limit: int = 50,
        token: TokenPayload = Depends(jwt_auth)
    ):
        """Get triage incidents with optional filtering."""
        
        try:
            # Parse filters
            status_filter = None
            if status:
                try:
                    status_filter = TriageStatus(status.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
            
            threat_filter = None
            if threat_level:
                try:
                    threat_filter = ThreatLevel(threat_level.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid threat level: {threat_level}")
            
            # Get incidents
            incidents = service_manager.emergency_triage.get_incidents(status_filter)
            
            # Apply threat level filter
            if threat_filter:
                incidents = [i for i in incidents if i.threat_level == threat_filter]
            
            # Apply limit
            incidents = incidents[:limit]
            
            # Convert to response format
            incident_data = []
            for incident in incidents:
                incident_dict = incident.to_dict()
                # Add computed fields
                incident_dict["age_seconds"] = (datetime.now() - incident.timestamp).total_seconds()
                incident_dict["is_active"] = incident.status not in [TriageStatus.RESOLVED, TriageStatus.FALSE_POSITIVE]
                incident_data.append(incident_dict)
            
            return APIResponse(
                data={
                    "incidents": incident_data,
                    "total_count": len(incident_data),
                    "filters_applied": {
                        "status": status,
                        "threat_level": threat_level,
                        "limit": limit
                    }
                },
                message=f"Retrieved {len(incident_data)} triage incidents"
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve triage incidents: {e}")
            raise HTTPException(status_code=500, detail=f"Incident retrieval failed: {str(e)}")
    
    
    @app.get("/v1/monitoring/triage/incident/{incident_id}", response_model=APIResponse)
    async def get_triage_incident(
        incident_id: str,
        token: TokenPayload = Depends(jwt_auth)
    ):
        """Get detailed information about a specific triage incident."""
        
        try:
            incident = service_manager.emergency_triage.incidents.get(incident_id)
            
            if not incident:
                raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
            
            incident_data = incident.to_dict()
            incident_data["age_seconds"] = (datetime.now() - incident.timestamp).total_seconds()
            incident_data["is_active"] = incident.status not in [TriageStatus.RESOLVED, TriageStatus.FALSE_POSITIVE]
            
            return APIResponse(
                data=incident_data,
                message=f"Retrieved triage incident {incident_id}"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve triage incident {incident_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Incident retrieval failed: {str(e)}")
    
    
    @app.post("/v1/monitoring/triage/incident/{incident_id}/status", response_model=APIResponse)
    async def update_triage_incident_status(
        incident_id: str,
        request: TriageStatusRequest,
        token: TokenPayload = Depends(jwt_auth)
    ):
        """Update the status of a triage incident."""
        
        try:
            incident = service_manager.emergency_triage.incidents.get(incident_id)
            
            if not incident:
                raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
            
            # Parse new status
            try:
                new_status = TriageStatus(request.new_status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {request.new_status}")
            
            # Update incident
            old_status = incident.status
            incident.status = new_status
            
            # Add response log entry
            incident.add_response_log(
                f"status_update_via_api",
                f"Status changed from {old_status.value} to {new_status.value}" + 
                (f". Notes: {request.notes}" if request.notes else "")
            )
            
            # Set resolution time if resolved
            if new_status in [TriageStatus.RESOLVED, TriageStatus.FALSE_POSITIVE]:
                incident.resolution_time = datetime.now()
            
            return APIResponse(
                data={
                    "incident_id": incident_id,
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "timestamp": datetime.now().isoformat()
                },
                message=f"Incident {incident_id} status updated to {new_status.value}"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update incident {incident_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")
    
    
    @app.get("/v1/monitoring/triage/statistics", response_model=APIResponse)
    async def get_triage_statistics(token: TokenPayload = Depends(jwt_auth)):
        """Get Emergency Triage System statistics."""
        
        try:
            stats = service_manager.emergency_triage.get_statistics()
            
            # Add real-time metrics
            active_by_level = {}
            for level in ThreatLevel:
                active_by_level[level.value] = len([
                    i for i in service_manager.emergency_triage.incidents.values()
                    if i.threat_level == level and i.status not in [TriageStatus.RESOLVED, TriageStatus.FALSE_POSITIVE]
                ])
            
            stats["active_incidents_by_threat_level"] = active_by_level
            stats["timestamp"] = datetime.now().isoformat()
            
            return APIResponse(
                data=stats,
                message="Emergency triage statistics retrieved"
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve triage statistics: {e}")
            raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")
    
    
    @app.post("/v1/monitoring/triage/test-incident", response_model=APIResponse)  
    async def create_test_incident(token: TokenPayload = Depends(jwt_auth)):
        """Create a test incident for system validation."""
        
        try:
            test_incident = service_manager.emergency_triage.detect_incident(
                source_component="test_system",
                incident_type="system_test",
                description="Test incident for archaeological integration validation",
                raw_data={
                    "test": True,
                    "integration": "archaeological_emergency_triage",
                    "metrics": {"cpu_usage": 95.0, "memory_usage": 85.0}
                },
                threat_level=ThreatLevel.LOW
            )
            
            return APIResponse(
                data={
                    "test_incident_id": test_incident.incident_id,
                    "status": test_incident.status.value,
                    "message": "Archaeological Emergency Triage System integration successful"
                },
                message="Test incident created successfully - Emergency Triage System operational"
            )
            
        except Exception as e:
            logger.error(f"Failed to create test incident: {e}")
            raise HTTPException(status_code=500, detail=f"Test incident creation failed: {str(e)}")