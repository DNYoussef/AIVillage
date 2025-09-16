"""
Unit tests for Phase 5 Training Loop components
Tests for core training functionality, optimization, and monitoring
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock training loop components
class MockTrainingConfig:
    def __init__(self):
        self.epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.max_grad_norm = 1.0
        self.use_amp = True
        self.gradient_accumulation_steps = 1
        self.log_interval = 50
        self.eval_interval = 500
        self.save_interval = 1000
        self.early_stopping_patience = 10

class MockTrainingState:
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.is_training = False
        self.start_time = None
        self.last_checkpoint_step = 0

class MockTrainingMetrics:
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.throughput = []
        self.memory_usage = []
    
    def update(self, loss, lr, grad_norm=None, throughput=None, memory=None):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        if grad_norm:
            self.gradient_norms.append(grad_norm)
        if throughput:
            self.throughput.append(throughput)
        if memory:
            self.memory_usage.append(memory)
    
    def get_average_loss(self, last_n=100):
        if not self.losses:
            return 0.0
        return sum(self.losses[-last_n:]) / min(len(self.losses), last_n)

class MockCheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoints = []
    
    def save_checkpoint(self, model, optimizer, scheduler, state, metrics):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{state.step}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'training_state': state.__dict__,
            'metrics': metrics.__dict__
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path)
    
    def get_latest_checkpoint(self):
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: os.path.getctime(x))

class MockMemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_history = []
    
    def update(self):
        # Simulate memory monitoring
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            self.peak_memory = max(self.peak_memory, self.current_memory)
        else:
            # Simulate CPU memory usage
            self.current_memory = np.random.randint(100, 1000)
            self.peak_memory = max(self.peak_memory, self.current_memory)
        
        self.memory_history.append(self.current_memory)
    
    def get_stats(self):
        return {
            'current_memory_mb': self.current_memory,
            'peak_memory_mb': self.peak_memory,
            'average_memory_mb': np.mean(self.memory_history) if self.memory_history else 0
        }

class MockEarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

class MockTrainingLoop:
    def __init__(self, model, config, device, checkpoint_dir):
        self.model = model
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        self.state = MockTrainingState()
        self.metrics = MockTrainingMetrics()
        self.checkpoint_manager = MockCheckpointManager(checkpoint_dir)
        self.memory_monitor = MockMemoryMonitor()
        self.early_stopping = MockEarlyStopping(config.early_stopping_patience)
        
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    def setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )
    
    def train_step(self, batch, loss_fn):
        self.model.train()
        
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        if self.config.use_amp and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
            
            self.scaler.scale(loss).backward()
            
            if self.config.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            else:
                grad_norm = None
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            if self.config.max_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            else:
                grad_norm = None
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update metrics
        self.metrics.update(
            loss=loss.item(),
            lr=self.optimizer.param_groups[0]['lr'],
            grad_norm=grad_norm.item() if grad_norm else None
        )
        
        self.state.step += 1
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'gradient_norm': grad_norm.item() if grad_norm else None
        }
    
    def validation_step(self, batch, loss_fn):
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
        
        return {
            'loss': loss.item(),
            'outputs': outputs,
            'targets': targets
        }
    
    def train(self, train_loader, loss_fn, val_loader=None):
        self.state.is_training = True
        self.state.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            
            # Training phase
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                step_metrics = self.train_step(batch, loss_fn)
                epoch_losses.append(step_metrics['loss'])
                
                # Memory monitoring
                self.memory_monitor.update()
                
                # Logging
                if self.state.step % self.config.log_interval == 0:
                    avg_loss = sum(epoch_losses[-self.config.log_interval:]) / min(len(epoch_losses), self.config.log_interval)
                    print(f"Step {self.state.step}, Loss: {avg_loss:.4f}")
                
                # Validation
                if val_loader and self.state.step % self.config.eval_interval == 0:
                    val_metrics = self.validate(val_loader, loss_fn)
                    
                    # Early stopping check
                    if self.early_stopping(val_metrics['accuracy']):
                        print(f"Early stopping at step {self.state.step}")
                        return
                
                # Checkpointing
                if self.state.step % self.config.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler, 
                        self.state, self.metrics
                    )
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
        
        self.state.is_training = False
    
    def validate(self, val_loader, loss_fn):
        self.model.eval()
        
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                val_metrics = self.validation_step(batch, loss_fn)
                val_losses.append(val_metrics['loss'])
                
                # Calculate accuracy
                outputs = val_metrics['outputs']
                targets = val_metrics['targets']
                
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total
        }

# Test cases
class TestTrainingConfig:
    """Test TrainingConfig functionality"""
    
    def test_config_initialization(self):
        """Test training configuration initialization"""
        config = MockTrainingConfig()
        
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-2
        assert config.max_grad_norm == 1.0
        assert config.use_amp == True
        assert config.gradient_accumulation_steps == 1
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = MockTrainingConfig()
        
        # Valid configuration
        assert config.epochs > 0
        assert config.learning_rate > 0
        assert config.weight_decay >= 0
        assert config.max_grad_norm > 0
        assert config.gradient_accumulation_steps >= 1

class TestTrainingState:
    """Test TrainingState functionality"""
    
    def test_state_initialization(self):
        """Test training state initialization"""
        state = MockTrainingState()
        
        assert state.epoch == 0
        assert state.step == 0
        assert state.best_metric == 0.0
        assert state.patience_counter == 0
        assert state.is_training == False
        assert state.start_time is None
    
    def test_state_updates(self):
        """Test training state updates"""
        state = MockTrainingState()
        
        # Update state
        state.epoch = 5
        state.step = 1000
        state.best_metric = 0.95
        state.is_training = True
        
        assert state.epoch == 5
        assert state.step == 1000
        assert state.best_metric == 0.95
        assert state.is_training == True

class TestTrainingMetrics:
    """Test TrainingMetrics functionality"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = MockTrainingMetrics()
        
        assert len(metrics.losses) == 0
        assert len(metrics.learning_rates) == 0
        assert len(metrics.gradient_norms) == 0
        assert len(metrics.throughput) == 0
        assert len(metrics.memory_usage) == 0
    
    def test_metrics_updates(self):
        """Test metrics updates"""
        metrics = MockTrainingMetrics()
        
        # Update metrics
        metrics.update(loss=0.5, lr=1e-4, grad_norm=0.1)
        metrics.update(loss=0.4, lr=9e-5, grad_norm=0.08)
        
        assert len(metrics.losses) == 2
        assert len(metrics.learning_rates) == 2
        assert len(metrics.gradient_norms) == 2
        assert metrics.losses[0] == 0.5
        assert metrics.losses[1] == 0.4
    
    def test_average_loss_calculation(self):
        """Test average loss calculation"""
        metrics = MockTrainingMetrics()
        
        # Add losses
        for i in range(10):
            metrics.update(loss=0.1 * i, lr=1e-4)
        
        avg_loss = metrics.get_average_loss(last_n=5)
        expected_avg = sum([0.1 * i for i in range(5, 10)]) / 5
        
        assert abs(avg_loss - expected_avg) < 1e-6

class TestCheckpointManager:
    """Test CheckpointManager functionality"""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation and saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MockCheckpointManager(temp_dir)
            
            # Create mock objects
            model = nn.Linear(10, 1)
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
            state = MockTrainingState()
            metrics = MockTrainingMetrics()
            
            state.step = 100
            
            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(model, optimizer, scheduler, state, metrics)
            
            assert os.path.exists(checkpoint_path)
            assert len(manager.checkpoints) == 1
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MockCheckpointManager(temp_dir)
            
            # Create and save checkpoint
            model = nn.Linear(10, 1)
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
            state = MockTrainingState()
            metrics = MockTrainingMetrics()
            
            state.step = 100
            checkpoint_path = manager.save_checkpoint(model, optimizer, scheduler, state, metrics)
            
            # Load checkpoint
            loaded_checkpoint = manager.load_checkpoint(checkpoint_path)
            
            assert 'model_state_dict' in loaded_checkpoint
            assert 'optimizer_state_dict' in loaded_checkpoint
            assert 'scheduler_state_dict' in loaded_checkpoint
            assert 'training_state' in loaded_checkpoint
            assert 'metrics' in loaded_checkpoint
    
    def test_latest_checkpoint_retrieval(self):
        """Test latest checkpoint retrieval"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MockCheckpointManager(temp_dir)
            
            model = nn.Linear(10, 1)
            optimizer = optim.Adam(model.parameters())
            state = MockTrainingState()
            metrics = MockTrainingMetrics()
            
            # Save multiple checkpoints
            for step in [100, 200, 300]:
                state.step = step
                manager.save_checkpoint(model, optimizer, None, state, metrics)
            
            latest_checkpoint = manager.get_latest_checkpoint()
            assert latest_checkpoint is not None
            assert "step_300" in str(latest_checkpoint)

class TestMemoryMonitor:
    """Test MemoryMonitor functionality"""
    
    def test_memory_monitoring_initialization(self):
        """Test memory monitor initialization"""
        monitor = MockMemoryMonitor()
        
        assert monitor.peak_memory == 0
        assert monitor.current_memory == 0
        assert len(monitor.memory_history) == 0
    
    def test_memory_monitoring_updates(self):
        """Test memory monitoring updates"""
        monitor = MockMemoryMonitor()
        
        # Update memory multiple times
        for _ in range(5):
            monitor.update()
        
        assert len(monitor.memory_history) == 5
        assert monitor.current_memory > 0
        assert monitor.peak_memory >= monitor.current_memory
    
    def test_memory_statistics(self):
        """Test memory statistics calculation"""
        monitor = MockMemoryMonitor()
        
        # Generate memory history
        for _ in range(10):
            monitor.update()
        
        stats = monitor.get_stats()
        
        assert 'current_memory_mb' in stats
        assert 'peak_memory_mb' in stats
        assert 'average_memory_mb' in stats
        assert stats['peak_memory_mb'] >= stats['current_memory_mb']
        assert stats['average_memory_mb'] > 0

class TestEarlyStopping:
    """Test EarlyStopping functionality"""
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization"""
        early_stopping = MockEarlyStopping(patience=5, min_delta=1e-3)
        
        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 1e-3
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert early_stopping.early_stop == False
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improving scores"""
        early_stopping = MockEarlyStopping(patience=3, min_delta=1e-3)
        
        # Improving scores should not trigger early stopping
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        for score in scores:
            early_stop = early_stopping(score)
            assert early_stop == False
        
        assert early_stopping.counter == 0
    
    def test_early_stopping_trigger(self):
        """Test early stopping trigger"""
        early_stopping = MockEarlyStopping(patience=2, min_delta=1e-3)
        
        # Initial improvement
        early_stopping(0.5)
        assert early_stopping.early_stop == False
        
        # No improvement for patience steps
        early_stopping(0.49)  # Counter = 1
        assert early_stopping.early_stop == False
        
        early_stopping(0.48)  # Counter = 2, should trigger
        assert early_stopping.early_stop == True

class TestTrainingLoop:
    """Test TrainingLoop functionality"""
    
    def test_training_loop_initialization(self):
        """Test training loop initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = nn.Linear(10, 1)
            config = MockTrainingConfig()
            device = torch.device('cpu')
            
            training_loop = MockTrainingLoop(model, config, device, temp_dir)
            
            assert training_loop.model == model
            assert training_loop.config == config
            assert training_loop.device == device
            assert training_loop.state.epoch == 0
            assert training_loop.state.step == 0
    
    def test_optimizer_setup(self):
        """Test optimizer setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = nn.Linear(10, 1)
            config = MockTrainingConfig()
            device = torch.device('cpu')
            
            training_loop = MockTrainingLoop(model, config, device, temp_dir)
            training_loop.setup_optimizer()
            
            assert training_loop.optimizer is not None
            assert training_loop.scheduler is not None
            assert isinstance(training_loop.optimizer, optim.AdamW)
    
    def test_train_step(self):
        """Test single training step"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = nn.Linear(10, 1)
            config = MockTrainingConfig()
            device = torch.device('cpu')
            
            training_loop = MockTrainingLoop(model, config, device, temp_dir)
            training_loop.setup_optimizer()
            
            # Create mock batch
            batch = (torch.randn(32, 10), torch.randn(32, 1))
            loss_fn = nn.MSELoss()
            
            # Perform training step
            metrics = training_loop.train_step(batch, loss_fn)
            
            assert 'loss' in metrics
            assert 'learning_rate' in metrics
            assert metrics['loss'] > 0
            assert training_loop.state.step == 1
    
    def test_validation_step(self):
        """Test single validation step"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = nn.Linear(10, 1)
            config = MockTrainingConfig()
            device = torch.device('cpu')
            
            training_loop = MockTrainingLoop(model, config, device, temp_dir)
            
            # Create mock batch
            batch = (torch.randn(32, 10), torch.randn(32, 1))
            loss_fn = nn.MSELoss()
            
            # Perform validation step
            metrics = training_loop.validation_step(batch, loss_fn)
            
            assert 'loss' in metrics
            assert 'outputs' in metrics
            assert 'targets' in metrics
            assert metrics['loss'] > 0

if __name__ == "__main__":
    pytest.main([__file__])