"""
Model Service - Manages model lifecycle and storage

This service is responsible for:
- Model CRUD operations and metadata management
- File system persistence and retrieval
- Model version control and handoff between phases
- Export functionality and format conversion
- Integration with training results

Size Target: <400 lines
"""

import hashlib
import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from interfaces.service_contracts import (
    IModelService, ModelMetadata, ModelHandoff, ModelExportRequest,
    ModelPhase, TaskStatus, Event, ModelSavedEvent, ModelHandoffCreatedEvent
)

logger = logging.getLogger(__name__)


class ModelService(IModelService):
    """Implementation of the Model Service."""
    
    def __init__(self, 
                 storage_path: str = "./models",
                 event_publisher=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.event_publisher = event_publisher
        
        # Create subdirectories for different phases
        for phase in ModelPhase:
            (self.storage_path / phase.value).mkdir(exist_ok=True)
        
        # In-memory model metadata cache
        self.models: Dict[str, ModelMetadata] = {}
        self.handoffs: Dict[str, ModelHandoff] = {}
        
        # Load existing models on startup
        asyncio.create_task(self._load_existing_models())
    
    async def _load_existing_models(self):
        """Load existing model metadata from storage."""
        try:
            metadata_file = self.storage_path / "models_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    for model_data in data.get('models', []):
                        model = ModelMetadata(**model_data)
                        self.models[model.model_id] = model
                logger.info(f"Loaded {len(self.models)} existing models")
        except Exception as e:
            logger.error(f"Failed to load existing models: {e}")
    
    async def _save_metadata(self):
        """Persist model metadata to storage."""
        try:
            metadata_file = self.storage_path / "models_metadata.json"
            data = {
                'models': [model.dict() for model in self.models.values()],
                'handoffs': [handoff.dict() for handoff in self.handoffs.values()],
                'updated_at': datetime.now().isoformat()
            }
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def save_model(self, metadata: ModelMetadata, model_data: bytes) -> str:
        """Save model and return model ID."""
        try:
            # Generate model ID if not provided
            if not metadata.model_id:
                metadata.model_id = str(uuid.uuid4())
            
            # Calculate file size and checksum
            metadata.file_size = len(model_data)
            metadata.checksum = hashlib.sha256(model_data).hexdigest()
            
            # Create file path
            phase_dir = self.storage_path / metadata.phase.value
            file_name = f"{metadata.name}_{metadata.version}_{metadata.model_id}.pth"
            file_path = phase_dir / file_name
            metadata.file_path = str(file_path)
            
            # Save model file
            with open(file_path, 'wb') as f:
                f.write(model_data)
            
            # Store metadata
            self.models[metadata.model_id] = metadata
            await self._save_metadata()
            
            # Publish model saved event
            if self.event_publisher:
                event = ModelSavedEvent(
                    source_service="model_service",
                    data={
                        "model_id": metadata.model_id,
                        "name": metadata.name,
                        "phase": metadata.phase.value
                    }
                )
                await self.event_publisher.publish(event)
            
            logger.info(f"Saved model {metadata.model_id} ({metadata.name})")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    async def load_model_data(self, model_id: str) -> Optional[bytes]:
        """Load model binary data."""
        model = await self.get_model(model_id)
        if not model:
            return None
        
        try:
            with open(model.file_path, 'rb') as f:
                data = f.read()
            
            # Verify checksum
            if hashlib.sha256(data).hexdigest() != model.checksum:
                logger.error(f"Checksum mismatch for model {model_id}")
                return None
            
            return data
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model.file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    async def list_models(self, phase: Optional[ModelPhase] = None) -> List[ModelMetadata]:
        """List models, optionally filtered by phase."""
        models = list(self.models.values())
        if phase:
            models = [model for model in models if model.phase == phase]
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    async def create_handoff(self, handoff: ModelHandoff) -> str:
        """Create model handoff between phases."""
        try:
            # Validate source model exists
            if handoff.model_id not in self.models:
                raise ValueError(f"Source model {handoff.model_id} not found")
            
            # Generate handoff ID
            if not handoff.handoff_id:
                handoff.handoff_id = str(uuid.uuid4())
            
            # Store handoff
            self.handoffs[handoff.handoff_id] = handoff
            await self._save_metadata()
            
            # Publish handoff event
            if self.event_publisher:
                event = ModelHandoffCreatedEvent(
                    source_service="model_service",
                    data={
                        "handoff_id": handoff.handoff_id,
                        "from_phase": handoff.from_phase.value,
                        "to_phase": handoff.to_phase.value,
                        "model_id": handoff.model_id
                    }
                )
                await self.event_publisher.publish(event)
            
            logger.info(f"Created handoff {handoff.handoff_id} from {handoff.from_phase} to {handoff.to_phase}")
            return handoff.handoff_id
            
        except Exception as e:
            logger.error(f"Failed to create handoff: {e}")
            raise
    
    async def export_models(self, request: ModelExportRequest) -> str:
        """Export models and return export path."""
        try:
            # Create export directory
            export_dir = self.storage_path / "exports" / str(uuid.uuid4())
            export_dir.mkdir(parents=True, exist_ok=True)
            
            exported_models = []
            
            for model_id in request.model_ids:
                model = await self.get_model(model_id)
                if not model:
                    logger.warning(f"Model {model_id} not found, skipping")
                    continue
                
                # Load model data
                model_data = await self.load_model_data(model_id)
                if not model_data:
                    logger.warning(f"Failed to load data for model {model_id}, skipping")
                    continue
                
                # Export model file
                export_filename = f"{model.name}_{model.version}.{request.export_format}"
                export_path = export_dir / export_filename
                
                with open(export_path, 'wb') as f:
                    f.write(model_data)
                
                # Include metadata if requested
                if request.include_metadata:
                    metadata_path = export_dir / f"{model.name}_{model.version}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(model.dict(), f, indent=2, default=str)
                
                exported_models.append({
                    "model_id": model_id,
                    "name": model.name,
                    "export_path": str(export_path)
                })
            
            # Create export manifest
            manifest = {
                "export_id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat(),
                "format": request.export_format,
                "models": exported_models,
                "total_models": len(exported_models)
            }
            
            manifest_path = export_dir / "export_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Compress if requested
            if request.compression:
                archive_path = export_dir.parent / f"{export_dir.name}.zip"
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in export_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(export_dir)
                            zipf.write(file_path, arcname)
                
                # Remove uncompressed directory
                shutil.rmtree(export_dir)
                final_path = str(archive_path)
            else:
                final_path = str(export_dir)
            
            logger.info(f"Exported {len(exported_models)} models to {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to export models: {e}")
            raise
    
    async def get_phase_winner(self, phase: ModelPhase) -> Optional[ModelMetadata]:
        """Get the winner model for a specific phase."""
        phase_models = await self.list_models(phase)
        winner_models = [model for model in phase_models if model.is_winner]
        return winner_models[0] if winner_models else None
    
    async def mark_as_winner(self, model_id: str) -> bool:
        """Mark a model as the winner for its phase."""
        model = await self.get_model(model_id)
        if not model:
            return False
        
        # Unmark previous winners in the same phase
        phase_models = await self.list_models(model.phase)
        for phase_model in phase_models:
            if phase_model.is_winner:
                phase_model.is_winner = False
        
        # Mark this model as winner
        model.is_winner = True
        await self._save_metadata()
        
        logger.info(f"Marked model {model_id} as winner for phase {model.phase}")
        return True
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and its associated files."""
        model = await self.get_model(model_id)
        if not model:
            return False
        
        try:
            # Delete model file
            Path(model.file_path).unlink(missing_ok=True)
            
            # Remove from metadata
            del self.models[model_id]
            await self._save_metadata()
            
            logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False


# Service factory
def create_model_service(storage_path: str = "./models", 
                        event_publisher=None) -> ModelService:
    """Create and configure the Model Service."""
    return ModelService(storage_path=storage_path, event_publisher=event_publisher)