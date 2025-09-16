"""
Base Phase Implementation

Provides a structured base class for all pipeline phases with
proper error handling, monitoring, and configuration management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging
import torch
import torch.nn as nn

from ...exceptions import PhaseExecutionError, handle_exception
from ...interfaces.base_interfaces import BasePhase, PhaseResult, BaseModel
from ...config.agent_forge_config import PhaseConfig

logger = logging.getLogger(__name__)


class EnhancedBasePhase(BasePhase):
    """
    Enhanced base phase with standardized error handling, monitoring,
    and configuration management.
    """
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        super().__init__(phase_name, config)
        self.logger = logging.getLogger(f"{__name__}.{phase_name}")
        self._phase_config: Optional[PhaseConfig] = None
        self._setup_phase_config()
        
    def _setup_phase_config(self):
        """Setup typed configuration from dict."""
        try:
            # This would be implemented by subclasses to create proper config objects
            pass
        except Exception as e:
            raise PhaseExecutionError(
                f"Failed to setup configuration for phase {self.phase_name}",
                phase_name=self.phase_name,
                details={"config": self.config, "error": str(e)}
            )
    
    def validate_config(self) -> None:
        """Validate phase configuration."""
        if not self.phase_name:
            raise PhaseExecutionError("Phase name is required")
        
        if not self.config:
            raise PhaseExecutionError(f"Configuration is required for phase {self.phase_name}")
        
        # Validate phase-specific configuration
        self._validate_phase_specific_config()
    
    @abstractmethod
    def _validate_phase_specific_config(self) -> None:
        """Validate phase-specific configuration. Implemented by subclasses."""
        pass
    
    async def run(self, model: BaseModel) -> PhaseResult:
        """Execute the phase with proper error handling and monitoring."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting phase: {self.phase_name}")
        
        try:
            # Validate configuration
            self.validate_config()
            
            # Validate model
            self._validate_model(model)
            
            # Execute pre-phase setup
            await self._pre_phase_setup(model)
            
            # Execute main phase logic
            result = await self._execute_phase(model)
            
            # Execute post-phase cleanup
            await self._post_phase_cleanup(model, result)
            
            self.end_time = datetime.now()
            duration = self.get_duration()
            
            self.logger.info(
                f"Phase {self.phase_name} completed successfully in {duration:.2f}s"
            )
            
            # Enhance result with timing information
            if hasattr(result, 'metrics') and result.metrics:
                result.metrics['duration_seconds'] = duration
                result.metrics['start_time'] = self.start_time.isoformat()
                result.metrics['end_time'] = self.end_time.isoformat()
            
            return result
            
        except PhaseExecutionError:
            # Re-raise phase execution errors
            raise
        except Exception as e:
            self.end_time = datetime.now()
            error_msg = f"Phase {self.phase_name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return PhaseResult(
                success=False,
                phase_name=self.phase_name,
                error=error_msg,
                metrics=self.metrics
            )
    
    def _validate_model(self, model: BaseModel) -> None:
        """Validate model state before phase execution."""
        if not model:
            raise PhaseExecutionError(
                f"Model is required for phase {self.phase_name}",
                phase_name=self.phase_name
            )
        
        if not model.is_loaded:
            raise PhaseExecutionError(
                f"Model must be loaded before executing phase {self.phase_name}",
                phase_name=self.phase_name
            )
    
    async def _pre_phase_setup(self, model: BaseModel) -> None:
        """Setup operations before phase execution. Override in subclasses."""
        self.logger.debug(f"Pre-phase setup for {self.phase_name}")
    
    async def _post_phase_cleanup(self, model: BaseModel, result: PhaseResult) -> None:
        """Cleanup operations after phase execution. Override in subclasses."""
        self.logger.debug(f"Post-phase cleanup for {self.phase_name}")
    
    @abstractmethod
    async def _execute_phase(self, model: BaseModel) -> PhaseResult:
        """Main phase execution logic. Implemented by subclasses."""
        pass
    
    def update_metrics(self, key: str, value: Any) -> None:
        """Update phase metrics safely."""
        self.metrics[key] = value
        self.logger.debug(f"Updated metric {key} = {value}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        return {
            "phase_name": self.phase_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": self.get_duration(),
            "metrics": self.metrics.copy()
        }


class ModelTrainingPhase(EnhancedBasePhase):
    """Base class for phases that involve model training."""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        super().__init__(phase_name, config)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.best_loss: float = float('inf')
        self.patience_counter: int = 0
    
    def _validate_model(self, model: BaseModel) -> None:
        """Enhanced model validation for training phases."""
        super()._validate_model(model)
        
        # Check if model is trainable
        if not hasattr(model.model, 'parameters'):
            raise PhaseExecutionError(
                f"Model does not have parameters for training in phase {self.phase_name}",
                phase_name=self.phase_name
            )
        
        # Check if model has trainable parameters
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise PhaseExecutionError(
                f"Model has no trainable parameters for phase {self.phase_name}",
                phase_name=self.phase_name
            )
        
        self.logger.info(f"Model has {trainable_params:,} trainable parameters")
    
    async def _pre_phase_setup(self, model: BaseModel) -> None:
        """Setup training components."""
        await super()._pre_phase_setup(model)
        
        try:
            self._setup_optimizer(model)
            self._setup_scheduler()
            self.logger.info("Training components initialized")
        except Exception as e:
            raise PhaseExecutionError(
                f"Failed to setup training components for phase {self.phase_name}",
                phase_name=self.phase_name,
                cause=e
            )
    
    def _setup_optimizer(self, model: BaseModel) -> None:
        """Setup optimizer. Override in subclasses for custom optimizers."""
        lr = self.config.get('learning_rate', 2e-5)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.logger.info(f"Setup AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler. Override in subclasses."""
        if self.optimizer and self.config.get('use_scheduler', True):
            scheduler_type = self.config.get('scheduler_type', 'linear')
            
            if scheduler_type == 'linear':
                from transformers import get_linear_schedule_with_warmup
                warmup_steps = self.config.get('warmup_steps', 500)
                total_steps = self.config.get('total_steps', 10000)
                
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                self.logger.info(f"Setup linear scheduler with warmup_steps={warmup_steps}")
    
    def should_early_stop(self, current_loss: float) -> bool:
        """Check if training should stop early."""
        patience = self.config.get('patience', 5)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience
    
    def save_checkpoint(self, model: BaseModel, step: int) -> None:
        """Save training checkpoint."""
        if 'checkpoint_dir' not in self.config:
            return
        
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{self.phase_name}-{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'metrics': self.metrics.copy(),
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")


class ModelEvaluationPhase(EnhancedBasePhase):
    """Base class for phases that involve model evaluation."""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        super().__init__(phase_name, config)
        self.evaluation_results: Dict[str, Any] = {}
    
    async def _execute_phase(self, model: BaseModel) -> PhaseResult:
        """Execute evaluation phase."""
        self.logger.info(f"Starting evaluation for phase {self.phase_name}")
        
        try:
            # Prepare evaluation data
            eval_data = await self._prepare_evaluation_data()
            
            # Run evaluation
            results = await self._evaluate_model(model, eval_data)
            
            # Process results
            processed_results = self._process_evaluation_results(results)
            
            self.evaluation_results = processed_results
            self.update_metrics('evaluation_results', processed_results)
            
            return PhaseResult(
                success=True,
                model=model,
                phase_name=self.phase_name,
                metrics=self.metrics,
                artifacts={'evaluation_results': processed_results}
            )
            
        except Exception as e:
            raise PhaseExecutionError(
                f"Evaluation failed in phase {self.phase_name}",
                phase_name=self.phase_name,
                cause=e
            )
    
    @abstractmethod
    async def _prepare_evaluation_data(self) -> Any:
        """Prepare data for evaluation. Implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _evaluate_model(self, model: BaseModel, eval_data: Any) -> Dict[str, Any]:
        """Evaluate model on prepared data. Implemented by subclasses."""
        pass
    
    def _process_evaluation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format evaluation results. Override in subclasses."""
        return results