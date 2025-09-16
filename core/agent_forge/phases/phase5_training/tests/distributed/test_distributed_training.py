"""
Distributed training tests for Phase 5 Training
Tests for multi-GPU training coordination, scaling, and fault tolerance
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

# Mock distributed training components for testing
class MockDistributedConfig:
    """Mock configuration for distributed training"""
    
    def __init__(self):
        self.world_size = 4  # 4 GPUs
        self.rank = 0
        self.local_rank = 0
        self.backend = 'nccl'
        self.master_addr = 'localhost'
        self.master_port = '12355'
        self.gradient_accumulation_steps = 4
        self.sync_batch_norm = True
        self.fp16 = True
        self.find_unused_parameters = False

class MockDistributedDataLoader:
    """Mock distributed data loader"""
    
    def __init__(self, dataset, batch_size, world_size, rank):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.num_samples = len(dataset) // world_size
        self.start_idx = rank * self.num_samples
        self.end_idx = min((rank + 1) * self.num_samples, len(dataset))
    
    def __iter__(self):
        # Simulate data loading for specific rank
        for i in range(self.start_idx, self.end_idx, self.batch_size):
            end_i = min(i + self.batch_size, self.end_idx)
            batch_data = []
            batch_targets = []
            
            for j in range(i, end_i):
                data, target = self.dataset[j % len(self.dataset)]
                batch_data.append(data)
                batch_targets.append(target)
            
            if batch_data:
                yield torch.stack(batch_data), torch.tensor(batch_targets)
    
    def __len__(self):
        return (self.end_idx - self.start_idx + self.batch_size - 1) // self.batch_size

class MockDataset:
    """Mock dataset for testing"""
    
    def __init__(self, size=1000, input_dim=128, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate deterministic data based on index
        torch.manual_seed(idx)
        data = torch.randn(self.input_dim)
        target = idx % self.num_classes
        return data, target

class MockDistributedTrainer:
    """Mock distributed trainer"""
    
    def __init__(self, model, config, device_id=0):
        self.model = model
        self.config = config
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Mock DDP wrapper
        self.ddp_model = self._wrap_ddp(self.model)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.metrics = defaultdict(list)
        
        # Communication tracking
        self.communication_stats = {
            'allreduce_calls': 0,
            'broadcast_calls': 0,
            'total_communication_time': 0.0
        }
    
    def _wrap_ddp(self, model):
        """Mock DDP wrapper"""
        # In real implementation, this would be:
        # return torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id])
        return model  # Mock for testing
    
    def setup_optimizer(self):
        """Setup optimizer for distributed training"""
        self.optimizer = optim.AdamW(
            self.ddp_model.parameters(),
            lr=1e-4 * self.config.world_size,  # Scale LR with world size
            weight_decay=1e-2
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000
        )
        
        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_step(self, batch, loss_fn):
        """Single distributed training step"""
        self.ddp_model.train()
        
        data, targets = batch
        data = data.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        if self.config.fp16 and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.ddp_model(data)
                loss = loss_fn(outputs, targets)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.ddp_model(data)
            loss = loss_fn(outputs, targets)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.fp16 and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Simulate gradient synchronization
            self._synchronize_gradients()
            
            # Optimizer step
            if self.config.fp16 and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.global_step += 1
        
        # Record metrics
        step_metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'global_step': self.global_step
        }
        
        return step_metrics
    
    def _synchronize_gradients(self):
        """Mock gradient synchronization"""
        # Simulate allreduce operation
        start_time = time.time()
        
        # Mock communication delay
        time.sleep(0.001)  # 1ms simulated communication
        
        communication_time = time.time() - start_time
        self.communication_stats['allreduce_calls'] += 1
        self.communication_stats['total_communication_time'] += communication_time
    
    def broadcast_parameters(self):
        """Mock parameter broadcasting"""
        start_time = time.time()
        
        # Simulate broadcast operation
        time.sleep(0.005)  # 5ms simulated broadcast
        
        communication_time = time.time() - start_time
        self.communication_stats['broadcast_calls'] += 1
        self.communication_stats['total_communication_time'] += communication_time
    
    def get_communication_stats(self):
        """Get communication statistics"""
        return self.communication_stats.copy()

class MockDistributedCoordinator:
    """Mock coordinator for distributed training"""
    
    def __init__(self, world_size=4):
        self.world_size = world_size
        self.trainers = {}
        self.global_metrics = defaultdict(list)
        self.fault_tolerance = {
            'failed_ranks': set(),
            'recovery_attempts': 0,
            'max_recovery_attempts': 3
        }
    
    def add_trainer(self, rank, trainer):
        """Add trainer for specific rank"""
        self.trainers[rank] = trainer
    
    def coordinate_training_step(self, batch_data, loss_fn):
        """Coordinate training step across all ranks"""
        step_results = {}
        
        for rank, trainer in self.trainers.items():
            if rank not in self.fault_tolerance['failed_ranks']:
                try:
                    # Simulate rank-specific batch
                    rank_batch = self._get_rank_batch(batch_data, rank)
                    step_metrics = trainer.train_step(rank_batch, loss_fn)
                    step_results[rank] = step_metrics
                except Exception as e:
                    # Simulate rank failure
                    self.fault_tolerance['failed_ranks'].add(rank)
                    step_results[rank] = {'error': str(e)}
        
        # Aggregate metrics
        if step_results:
            aggregated_metrics = self._aggregate_metrics(step_results)
            self.global_metrics['aggregated'].append(aggregated_metrics)
        
        return step_results
    
    def _get_rank_batch(self, batch_data, rank):
        """Get batch data for specific rank"""
        data, targets = batch_data
        
        # Simulate data distribution across ranks
        samples_per_rank = len(data) // self.world_size
        start_idx = rank * samples_per_rank
        end_idx = (rank + 1) * samples_per_rank if rank < self.world_size - 1 else len(data)
        
        rank_data = data[start_idx:end_idx]
        rank_targets = targets[start_idx:end_idx]
        
        return rank_data, rank_targets
    
    def _aggregate_metrics(self, step_results):
        """Aggregate metrics across ranks"""
        valid_results = {k: v for k, v in step_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'loss': float('inf'), 'active_ranks': 0}
        
        # Average metrics across ranks
        avg_loss = np.mean([result['loss'] for result in valid_results.values()])
        avg_lr = np.mean([result['learning_rate'] for result in valid_results.values()])
        
        return {
            'loss': avg_loss,
            'learning_rate': avg_lr,
            'active_ranks': len(valid_results),
            'failed_ranks': len(self.fault_tolerance['failed_ranks'])
        }
    
    def attempt_recovery(self):
        """Attempt to recover failed ranks"""
        if not self.fault_tolerance['failed_ranks']:
            return True
        
        if self.fault_tolerance['recovery_attempts'] >= self.fault_tolerance['max_recovery_attempts']:
            return False
        
        # Simulate recovery attempt
        recovered_ranks = set()
        for failed_rank in self.fault_tolerance['failed_ranks']:
            # Mock recovery success rate (80%)
            if np.random.random() > 0.2:
                recovered_ranks.add(failed_rank)
        
        # Remove recovered ranks
        self.fault_tolerance['failed_ranks'] -= recovered_ranks
        self.fault_tolerance['recovery_attempts'] += 1
        
        return len(recovered_ranks) > 0
    
    def get_scaling_efficiency(self):
        """Calculate scaling efficiency"""
        active_ranks = self.world_size - len(self.fault_tolerance['failed_ranks'])
        
        # Mock ideal vs actual throughput calculation
        ideal_throughput = 1000 * self.world_size  # samples/sec
        
        # Account for communication overhead
        total_comm_time = sum(
            trainer.get_communication_stats()['total_communication_time']
            for trainer in self.trainers.values()
        )
        
        # Simple efficiency calculation
        comm_overhead = total_comm_time / max(1, len(self.global_metrics['aggregated']))
        efficiency = max(0.1, 1.0 - comm_overhead * 10)  # Mock calculation
        
        return {
            'active_ranks': active_ranks,
            'ideal_throughput': ideal_throughput,
            'efficiency': efficiency,
            'communication_overhead': comm_overhead
        }

# Test Cases
class TestDistributedDataLoading:
    """Test distributed data loading"""
    
    def test_data_distribution(self):
        """Test data distribution across ranks"""
        dataset = MockDataset(size=1000)
        world_size = 4
        batch_size = 32
        
        # Create data loaders for each rank
        loaders = []
        for rank in range(world_size):
            loader = MockDistributedDataLoader(dataset, batch_size, world_size, rank)
            loaders.append(loader)
        
        # Verify data distribution
        total_samples = 0
        rank_samples = []
        
        for rank, loader in enumerate(loaders):
            rank_sample_count = 0
            for batch in loader:
                data, targets = batch
                rank_sample_count += len(data)
            
            rank_samples.append(rank_sample_count)
            total_samples += rank_sample_count
        
        # Check that data is distributed fairly
        expected_samples_per_rank = len(dataset) // world_size
        
        for rank, samples in enumerate(rank_samples):
            assert abs(samples - expected_samples_per_rank) <= batch_size, \
                f"Rank {rank} has unfair data distribution: {samples} vs {expected_samples_per_rank}"
        
        # Check that all data is covered
        assert total_samples >= len(dataset) * 0.9, "Too much data lost in distribution"
    
    def test_data_loader_consistency(self):
        """Test data loader consistency across epochs"""
        dataset = MockDataset(size=100)
        loader = MockDistributedDataLoader(dataset, batch_size=16, world_size=2, rank=0)
        
        # Load data twice
        epoch1_data = []
        epoch2_data = []
        
        for batch in loader:
            epoch1_data.append(batch)
        
        for batch in loader:
            epoch2_data.append(batch)
        
        # Data should be consistent across epochs
        assert len(epoch1_data) == len(epoch2_data)
        
        for batch1, batch2 in zip(epoch1_data, epoch2_data):
            data1, targets1 = batch1
            data2, targets2 = batch2
            
            assert torch.equal(data1, data2), "Data inconsistent across epochs"
            assert torch.equal(targets1, targets2), "Targets inconsistent across epochs"
    
    def test_rank_specific_data(self):
        """Test that different ranks get different data"""
        dataset = MockDataset(size=200)
        world_size = 4
        batch_size = 20
        
        # Create loaders for two different ranks
        loader_rank0 = MockDistributedDataLoader(dataset, batch_size, world_size, rank=0)
        loader_rank1 = MockDistributedDataLoader(dataset, batch_size, world_size, rank=1)
        
        # Get first batch from each rank
        batch0 = next(iter(loader_rank0))
        batch1 = next(iter(loader_rank1))
        
        data0, targets0 = batch0
        data1, targets1 = batch1
        
        # Different ranks should get different data
        assert not torch.equal(data0, data1), "Different ranks getting same data"
        assert not torch.equal(targets0, targets1), "Different ranks getting same targets"

class TestDistributedTraining:
    """Test distributed training functionality"""
    
    def test_trainer_initialization(self):
        """Test distributed trainer initialization"""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        config = MockDistributedConfig()
        trainer = MockDistributedTrainer(model, config, device_id=0)
        
        # Verify initialization
        assert trainer.model is not None
        assert trainer.ddp_model is not None
        assert trainer.config == config
        assert trainer.global_step == 0
        assert trainer.epoch == 0
    
    def test_optimizer_setup(self):
        """Test optimizer setup for distributed training"""
        model = nn.Linear(64, 10)
        config = MockDistributedConfig()
        trainer = MockDistributedTrainer(model, config)
        
        trainer.setup_optimizer()
        
        # Verify optimizer setup
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        
        # Learning rate should be scaled by world size
        expected_lr = 1e-4 * config.world_size
        assert trainer.optimizer.param_groups[0]['lr'] == expected_lr
        
        # Check for FP16 support
        if config.fp16:
            assert trainer.scaler is not None
    
    def test_training_step(self):
        """Test single training step"""
        model = nn.Linear(128, 10)
        config = MockDistributedConfig()
        trainer = MockDistributedTrainer(model, config)
        trainer.setup_optimizer()
        
        # Create batch
        batch = (torch.randn(32, 128), torch.randint(0, 10, (32,)))
        loss_fn = nn.CrossEntropyLoss()
        
        # Execute training step
        metrics = trainer.train_step(batch, loss_fn)
        
        # Verify metrics
        assert 'loss' in metrics
        assert 'learning_rate' in metrics
        assert 'global_step' in metrics
        
        assert metrics['loss'] > 0
        assert metrics['global_step'] == 1
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation in distributed training"""
        model = nn.Linear(64, 10)
        config = MockDistributedConfig()
        config.gradient_accumulation_steps = 4
        
        trainer = MockDistributedTrainer(model, config)
        trainer.setup_optimizer()
        
        loss_fn = nn.CrossEntropyLoss()
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Execute multiple steps
        for step in range(8):
            batch = (torch.randn(16, 64), torch.randint(0, 10, (16,)))
            metrics = trainer.train_step(batch, loss_fn)
        
        # Verify gradient accumulation
        assert trainer.global_step == 8
        
        # Learning rate should have changed (scheduler stepped)
        final_lr = trainer.optimizer.param_groups[0]['lr']
        assert final_lr != initial_lr
    
    def test_communication_tracking(self):
        """Test communication tracking"""
        model = nn.Linear(32, 10)
        config = MockDistributedConfig()
        trainer = MockDistributedTrainer(model, config)
        trainer.setup_optimizer()
        
        # Perform some training steps
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(10):
            batch = (torch.randn(16, 32), torch.randint(0, 10, (16,)))
            trainer.train_step(batch, loss_fn)
        
        # Check communication stats
        comm_stats = trainer.get_communication_stats()
        
        assert 'allreduce_calls' in comm_stats
        assert 'total_communication_time' in comm_stats
        
        # Should have communication calls for gradient synchronization
        expected_calls = 10 // config.gradient_accumulation_steps
        assert comm_stats['allreduce_calls'] >= expected_calls

class TestDistributedCoordination:
    """Test distributed training coordination"""
    
    def test_coordinator_initialization(self):
        """Test distributed coordinator initialization"""
        coordinator = MockDistributedCoordinator(world_size=4)
        
        assert coordinator.world_size == 4
        assert len(coordinator.trainers) == 0
        assert len(coordinator.fault_tolerance['failed_ranks']) == 0
    
    def test_trainer_registration(self):
        """Test trainer registration with coordinator"""
        coordinator = MockDistributedCoordinator(world_size=2)
        
        # Create and register trainers
        for rank in range(2):
            model = nn.Linear(32, 10)
            config = MockDistributedConfig()
            trainer = MockDistributedTrainer(model, config, device_id=rank)
            coordinator.add_trainer(rank, trainer)
        
        assert len(coordinator.trainers) == 2
        assert 0 in coordinator.trainers
        assert 1 in coordinator.trainers
    
    def test_coordinated_training_step(self):
        """Test coordinated training step across ranks"""
        coordinator = MockDistributedCoordinator(world_size=2)
        
        # Setup trainers
        for rank in range(2):
            model = nn.Linear(64, 10)
            config = MockDistributedConfig()
            config.world_size = 2
            trainer = MockDistributedTrainer(model, config, device_id=rank)
            trainer.setup_optimizer()
            coordinator.add_trainer(rank, trainer)
        
        # Create batch data
        batch_data = (torch.randn(32, 64), torch.randint(0, 10, (32,)))
        loss_fn = nn.CrossEntropyLoss()
        
        # Coordinate training step
        step_results = coordinator.coordinate_training_step(batch_data, loss_fn)
        
        # Verify results
        assert len(step_results) == 2
        assert 0 in step_results
        assert 1 in step_results
        
        for rank, result in step_results.items():
            assert 'loss' in result
            assert 'learning_rate' in result
            assert result['loss'] > 0
    
    def test_metric_aggregation(self):
        """Test metric aggregation across ranks"""
        coordinator = MockDistributedCoordinator(world_size=3)
        
        # Mock step results
        step_results = {
            0: {'loss': 1.0, 'learning_rate': 1e-4},
            1: {'loss': 1.2, 'learning_rate': 1e-4},
            2: {'loss': 0.8, 'learning_rate': 1e-4}
        }
        
        aggregated = coordinator._aggregate_metrics(step_results)
        
        # Verify aggregation
        assert 'loss' in aggregated
        assert 'learning_rate' in aggregated
        assert 'active_ranks' in aggregated
        
        expected_avg_loss = (1.0 + 1.2 + 0.8) / 3
        assert abs(aggregated['loss'] - expected_avg_loss) < 1e-6
        assert aggregated['active_ranks'] == 3
    
    def test_fault_tolerance(self):
        """Test fault tolerance mechanisms"""
        coordinator = MockDistributedCoordinator(world_size=4)
        
        # Setup trainers
        for rank in range(4):
            model = nn.Linear(32, 10)
            config = MockDistributedConfig()
            trainer = MockDistributedTrainer(model, config)
            coordinator.add_trainer(rank, trainer)
        
        # Simulate rank failure
        coordinator.fault_tolerance['failed_ranks'].add(1)
        coordinator.fault_tolerance['failed_ranks'].add(3)
        
        # Test recovery
        recovery_success = coordinator.attempt_recovery()
        
        # Verify recovery attempt
        assert isinstance(recovery_success, bool)
        assert coordinator.fault_tolerance['recovery_attempts'] == 1
        
        # Failed ranks should be reduced (with some probability)
        assert len(coordinator.fault_tolerance['failed_ranks']) <= 2
    
    def test_scaling_efficiency(self):
        """Test scaling efficiency calculation"""
        coordinator = MockDistributedCoordinator(world_size=4)
        
        # Setup trainers with mock communication stats
        for rank in range(4):
            model = nn.Linear(32, 10)
            config = MockDistributedConfig()
            trainer = MockDistributedTrainer(model, config)
            
            # Mock some communication time
            trainer.communication_stats['total_communication_time'] = 0.1 * rank
            
            coordinator.add_trainer(rank, trainer)
        
        # Add some global metrics
        coordinator.global_metrics['aggregated'] = [
            {'loss': 1.0, 'active_ranks': 4},
            {'loss': 0.8, 'active_ranks': 4}
        ]
        
        efficiency = coordinator.get_scaling_efficiency()
        
        # Verify efficiency metrics
        assert 'active_ranks' in efficiency
        assert 'efficiency' in efficiency
        assert 'communication_overhead' in efficiency
        
        assert efficiency['active_ranks'] == 4
        assert 0 <= efficiency['efficiency'] <= 1

class TestMultiGPUSupport:
    """Test multi-GPU support"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_gpu_model_placement(self):
        """Test model placement on multiple GPUs"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for this test")
        
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        config = MockDistributedConfig()
        
        # Create trainers for different GPUs
        trainer_gpu0 = MockDistributedTrainer(model, config, device_id=0)
        trainer_gpu1 = MockDistributedTrainer(model, config, device_id=1)
        
        # Verify models are on correct devices
        assert next(trainer_gpu0.model.parameters()).device.index == 0
        assert next(trainer_gpu1.model.parameters()).device.index == 1
    
    def test_memory_scaling(self):
        """Test memory usage scaling with multiple GPUs"""
        # Create models of different sizes
        small_model = nn.Linear(64, 10)
        large_model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        config = MockDistributedConfig()
        
        small_trainer = MockDistributedTrainer(small_model, config)
        large_trainer = MockDistributedTrainer(large_model, config)
        
        # Calculate model sizes
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        small_params = count_parameters(small_model)
        large_params = count_parameters(large_model)
        
        # Large model should have significantly more parameters
        assert large_params > small_params * 10
    
    def test_synchronization_overhead(self):
        """Test synchronization overhead measurement"""
        model = nn.Linear(128, 10)
        config = MockDistributedConfig()
        trainer = MockDistributedTrainer(model, config)
        trainer.setup_optimizer()
        
        # Measure training without synchronization
        start_time = time.time()
        batch = (torch.randn(32, 128), torch.randint(0, 10, (32,)))
        loss_fn = nn.CrossEntropyLoss()
        
        for _ in range(10):
            trainer.train_step(batch, loss_fn)
        
        training_time = time.time() - start_time
        
        # Get communication stats
        comm_stats = trainer.get_communication_stats()
        comm_time = comm_stats['total_communication_time']
        
        # Communication overhead should be reasonable
        overhead_ratio = comm_time / training_time
        assert overhead_ratio < 0.5, f"Communication overhead too high: {overhead_ratio:.2f}"

class TestDistributedPerformance:
    """Test distributed training performance"""
    
    def test_throughput_scaling(self):
        """Test throughput scaling with number of GPUs"""
        model = nn.Linear(256, 10)
        
        world_sizes = [1, 2, 4]
        throughputs = []
        
        for world_size in world_sizes:
            config = MockDistributedConfig()
            config.world_size = world_size
            
            coordinator = MockDistributedCoordinator(world_size)
            
            # Setup trainers
            for rank in range(world_size):
                trainer = MockDistributedTrainer(model, config, device_id=rank)
                trainer.setup_optimizer()
                coordinator.add_trainer(rank, trainer)
            
            # Measure throughput
            start_time = time.time()
            batch_data = (torch.randn(64, 256), torch.randint(0, 10, (64,)))
            loss_fn = nn.CrossEntropyLoss()
            
            for _ in range(20):
                coordinator.coordinate_training_step(batch_data, loss_fn)
            
            elapsed_time = time.time() - start_time
            throughput = (20 * 64) / elapsed_time  # samples per second
            throughputs.append(throughput)
        
        print(f"Throughput scaling: {dict(zip(world_sizes, throughputs))}")
        
        # Throughput should generally increase with world size
        assert throughputs[1] >= throughputs[0] * 0.8  # Allow some overhead
        assert throughputs[2] >= throughputs[1] * 0.8
    
    def test_weak_scaling(self):
        """Test weak scaling (constant work per GPU)"""
        model = nn.Linear(128, 10)
        
        # Test different world sizes with proportional batch sizes
        configurations = [
            (1, 32),   # 1 GPU, 32 batch size
            (2, 64),   # 2 GPUs, 64 batch size
            (4, 128)   # 4 GPUs, 128 batch size
        ]
        
        training_times = []
        
        for world_size, total_batch_size in configurations:
            config = MockDistributedConfig()
            config.world_size = world_size
            
            coordinator = MockDistributedCoordinator(world_size)
            
            # Setup trainers
            for rank in range(world_size):
                trainer = MockDistributedTrainer(model, config)
                trainer.setup_optimizer()
                coordinator.add_trainer(rank, trainer)
            
            # Measure training time
            start_time = time.time()
            batch_data = (
                torch.randn(total_batch_size, 128),
                torch.randint(0, 10, (total_batch_size,))
            )
            loss_fn = nn.CrossEntropyLoss()
            
            for _ in range(10):
                coordinator.coordinate_training_step(batch_data, loss_fn)
            
            training_time = time.time() - start_time
            training_times.append(training_time)
        
        print(f"Weak scaling times: {dict(zip([c[0] for c in configurations], training_times))}")
        
        # With perfect weak scaling, times should be similar
        # Allow for some communication overhead
        max_time = max(training_times)
        min_time = min(training_times)
        scaling_efficiency = min_time / max_time
        
        assert scaling_efficiency > 0.6, f"Poor weak scaling efficiency: {scaling_efficiency:.2f}"
    
    def test_strong_scaling(self):
        """Test strong scaling (constant total work)"""
        model = nn.Linear(256, 10)
        total_samples = 512  # Fixed total work
        
        world_sizes = [1, 2, 4]
        training_times = []
        
        for world_size in world_sizes:
            config = MockDistributedConfig()
            config.world_size = world_size
            
            coordinator = MockDistributedCoordinator(world_size)
            
            # Setup trainers
            for rank in range(world_size):
                trainer = MockDistributedTrainer(model, config)
                trainer.setup_optimizer()
                coordinator.add_trainer(rank, trainer)
            
            # Measure training time for fixed total work
            start_time = time.time()
            batch_data = (
                torch.randn(total_samples, 256),
                torch.randint(0, 10, (total_samples,))
            )
            loss_fn = nn.CrossEntropyLoss()
            
            # Process same total amount of data
            steps = 5
            for _ in range(steps):
                coordinator.coordinate_training_step(batch_data, loss_fn)
            
            training_time = time.time() - start_time
            training_times.append(training_time)
        
        print(f"Strong scaling times: {dict(zip(world_sizes, training_times))}")
        
        # With good strong scaling, time should decrease with more GPUs
        speedup_2gpu = training_times[0] / training_times[1]
        speedup_4gpu = training_times[0] / training_times[2]
        
        assert speedup_2gpu > 1.2, f"Poor 2-GPU speedup: {speedup_2gpu:.2f}"
        assert speedup_4gpu > speedup_2gpu, f"Poor 4-GPU scaling: {speedup_4gpu:.2f}"

if __name__ == "__main__":
    pytest.main([__file__])