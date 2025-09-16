"""
Agent Forge Phase 5: Distributed Training System
================================================

Multi-GPU distributed training coordination with fault tolerance,
load balancing, and optimal resource utilization.

Key Features:
- PyTorch DistributedDataParallel (DDP)
- Fault-tolerant training
- Dynamic GPU allocation
- Cross-node synchronization
- Memory optimization
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import socket
import subprocess
from typing import Dict, List, Optional, Callable, Any
import logging
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DistributedConfig:
    """Configuration for distributed training setup."""
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    timeout_minutes: int = 30
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25


class DistributedTrainer:
    """
    Distributed training coordinator for Agent Forge Phase 5.

    Handles multi-GPU training setup, process coordination, and fault tolerance
    for efficient BitNet model training.
    """

    def __init__(self, config, device_ids: Optional[List[int]] = None):
        self.config = config
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.world_size = len(self.device_ids)

        # Distributed configuration
        self.dist_config = DistributedConfig(
            world_size=self.world_size,
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

        # Setup logging
        self.logger = self._setup_logging()

        # Training state tracking
        self.is_initialized = False
        self.process_group = None
        self.rank = 0
        self.local_rank = 0

        # Performance monitoring
        self.sync_times = []
        self.communication_overhead = []
        self.gpu_utilization = {}

        self.logger.info(f"Distributed trainer initialized for {self.world_size} devices")

    def _setup_logging(self) -> logging.Logger:
        """Setup distributed training specific logging."""
        logger = logging.getLogger(f'distributed_trainer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def find_free_port(self) -> str:
        """Find a free port for distributed communication."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return str(port)

    def setup_environment(self) -> None:
        """Setup environment variables for distributed training."""
        # Set master address and port
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = self.dist_config.master_addr

        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = self.find_free_port()

        # Set world size and rank
        os.environ["WORLD_SIZE"] = str(self.dist_config.world_size)

        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        self.logger.info(f"Environment setup - Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    def initialize_process_group(self) -> bool:
        """Initialize distributed process group with fault tolerance."""
        try:
            # Setup environment
            self.setup_environment()

            # Get rank information
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.dist_config.backend,
                    init_method=self.dist_config.init_method,
                    world_size=self.dist_config.world_size,
                    rank=self.rank,
                    timeout=torch.distributed.default_pg_timeout
                )

            # Verify initialization
            if dist.is_initialized():
                self.is_initialized = True
                self.process_group = dist.group.WORLD

                # Synchronize all processes
                dist.barrier()

                self.logger.info(
                    f"Process group initialized - Rank: {self.rank}/{self.world_size}, "
                    f"Local rank: {self.local_rank}, Backend: {self.dist_config.backend}"
                )

                return True
            else:
                self.logger.error("Failed to initialize process group")
                return False

        except Exception as e:
            self.logger.error(f"Process group initialization failed: {e}")
            return False

    def setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup model for distributed training with DDP."""
        if not self.is_initialized:
            raise RuntimeError("Process group not initialized")

        try:
            # Move model to correct device
            device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Wrap with DistributedDataParallel
            if self.world_size > 1:
                model = DDP(
                    model,
                    device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                    output_device=self.local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=self.dist_config.find_unused_parameters,
                    bucket_cap_mb=self.dist_config.bucket_cap_mb
                )

                self.logger.info(f"Model wrapped with DDP on device {device}")
            else:
                self.logger.info(f"Single device training on {device}")

            return model

        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise

    def synchronize_processes(self) -> float:
        """Synchronize all processes and measure communication overhead."""
        if not self.is_initialized or self.world_size == 1:
            return 0.0

        start_time = time.time()

        try:
            dist.barrier()
            sync_time = time.time() - start_time
            self.sync_times.append(sync_time)

            if len(self.sync_times) % 100 == 0:  # Log every 100 synchronizations
                avg_sync_time = sum(self.sync_times[-100:]) / min(100, len(self.sync_times))
                self.logger.debug(f"Average sync time (last 100): {avg_sync_time:.4f}s")

            return sync_time

        except Exception as e:
            self.logger.error(f"Process synchronization failed: {e}")
            return float('inf')

    def all_reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Perform all-reduce operation on tensor across all processes."""
        if not self.is_initialized or self.world_size == 1:
            return tensor

        try:
            # Ensure tensor is on correct device
            if torch.cuda.is_available() and not tensor.is_cuda:
                tensor = tensor.cuda(self.local_rank)

            # Perform all-reduce
            dist.all_reduce(tensor, op=op)

            # Average for sum operations
            if op == dist.ReduceOp.SUM:
                tensor = tensor / self.world_size

            return tensor

        except Exception as e:
            self.logger.error(f"All-reduce operation failed: {e}")
            return tensor

    def all_gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes."""
        if not self.is_initialized or self.world_size == 1:
            return [tensor]

        try:
            # Ensure tensor is on correct device
            if torch.cuda.is_available() and not tensor.is_cuda:
                tensor = tensor.cuda(self.local_rank)

            # Prepare output list
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]

            # All-gather operation
            dist.all_gather(gathered_tensors, tensor)

            return gathered_tensors

        except Exception as e:
            self.logger.error(f"All-gather operation failed: {e}")
            return [tensor]

    def broadcast_object(self, obj: Any, src_rank: int = 0) -> Any:
        """Broadcast object from source rank to all processes."""
        if not self.is_initialized or self.world_size == 1:
            return obj

        try:
            # Create object list for broadcast
            object_list = [obj if self.rank == src_rank else None]

            # Broadcast object
            dist.broadcast_object_list(object_list, src=src_rank)

            return object_list[0]

        except Exception as e:
            self.logger.error(f"Object broadcast failed: {e}")
            return obj

    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Reduce metrics across all processes."""
        if not self.is_initialized or self.world_size == 1:
            return metrics

        reduced_metrics = {}

        for key, value in metrics.items():
            try:
                # Convert to tensor
                tensor = torch.tensor(value, dtype=torch.float32)

                # Reduce across processes
                reduced_tensor = self.all_reduce_tensor(tensor)

                # Convert back to float
                reduced_metrics[key] = reduced_tensor.item()

            except Exception as e:
                self.logger.error(f"Failed to reduce metric {key}: {e}")
                reduced_metrics[key] = value

        return reduced_metrics

    def save_distributed_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        checkpoint_dir: str
    ) -> bool:
        """Save checkpoint from rank 0 only."""
        if self.rank != 0:
            return True  # Non-master ranks don't save

        try:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract model state dict (handle DDP wrapper)
            model_state_dict = (
                model.module.state_dict() if hasattr(model, 'module')
                else model.state_dict()
            )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'distributed_config': {
                    'world_size': self.world_size,
                    'backend': self.dist_config.backend
                }
            }

            torch.save(checkpoint, checkpoint_path)

            self.logger.info(f"Distributed checkpoint saved: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save distributed checkpoint: {e}")
            return False

    def load_distributed_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_path: str
    ) -> int:
        """Load checkpoint for distributed training."""
        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
            )

            # Load model state dict (handle DDP wrapper)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer and scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            epoch = checkpoint['epoch']

            # Synchronize all processes
            self.synchronize_processes()

            self.logger.info(f"Distributed checkpoint loaded from epoch {epoch}")
            return epoch

        except Exception as e:
            self.logger.error(f"Failed to load distributed checkpoint: {e}")
            return 0

    def monitor_gpu_utilization(self) -> Dict[str, float]:
        """Monitor GPU utilization across all devices."""
        if not torch.cuda.is_available():
            return {}

        utilization = {}

        try:
            for i, device_id in enumerate(self.device_ids):
                # Get GPU memory usage
                memory_allocated = torch.cuda.memory_allocated(device_id)
                memory_reserved = torch.cuda.memory_reserved(device_id)
                max_memory = torch.cuda.get_device_properties(device_id).total_memory

                utilization[f'gpu_{device_id}'] = {
                    'memory_allocated_gb': memory_allocated / (1024**3),
                    'memory_reserved_gb': memory_reserved / (1024**3),
                    'memory_utilization_pct': (memory_allocated / max_memory) * 100,
                    'max_memory_gb': max_memory / (1024**3)
                }

            # All-gather utilization data from all processes
            if self.is_initialized and self.world_size > 1:
                all_utilization = self.all_gather_tensors(
                    torch.tensor(list(utilization.values()), dtype=torch.float32)
                )
                # Combine utilization data from all processes
                for rank, rank_util in enumerate(all_utilization):
                    utilization[f'rank_{rank}_util'] = rank_util.tolist()

        except Exception as e:
            self.logger.error(f"GPU utilization monitoring failed: {e}")

        return utilization

    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        try:
            if self.is_initialized and dist.is_initialized():
                # Final synchronization
                dist.barrier()

                # Destroy process group
                dist.destroy_process_group()

                self.logger.info("Distributed training cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

        finally:
            self.is_initialized = False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get distributed training performance statistics."""
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.dist_config.backend,
            'sync_times': {
                'count': len(self.sync_times),
                'average': sum(self.sync_times) / len(self.sync_times) if self.sync_times else 0,
                'max': max(self.sync_times) if self.sync_times else 0,
                'min': min(self.sync_times) if self.sync_times else 0
            },
            'gpu_utilization': self.gpu_utilization,
            'communication_overhead': sum(self.communication_overhead),
            'is_initialized': self.is_initialized
        }


def run_distributed_training(
    rank: int,
    world_size: int,
    training_function: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Helper function to run distributed training across multiple processes.

    Args:
        rank: Process rank
        world_size: Total number of processes
        training_function: Function to execute in each process
        *args, **kwargs: Arguments for training function
    """

    # Setup environment for this process
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    try:
        # Run training function
        result = training_function(rank, world_size, *args, **kwargs)
        return result

    except Exception as e:
        print(f"Training process {rank} failed: {e}")
        raise

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # Example distributed training setup
    def test_distributed_setup():
        """Test distributed training setup."""
        from training_config import TrainingConfig

        config = TrainingConfig()
        device_ids = list(range(min(2, torch.cuda.device_count())))

        # Create distributed trainer
        trainer = DistributedTrainer(config, device_ids)

        # Test initialization
        if trainer.initialize_process_group():
            print("✓ Process group initialized successfully")

            # Test model setup
            model = torch.nn.Linear(10, 5)
            distributed_model = trainer.setup_model(model)
            print(f"✓ Model setup complete: {type(distributed_model)}")

            # Test synchronization
            sync_time = trainer.synchronize_processes()
            print(f"✓ Process synchronization: {sync_time:.4f}s")

            # Test GPU monitoring
            gpu_stats = trainer.monitor_gpu_utilization()
            print(f"✓ GPU monitoring: {len(gpu_stats)} devices")

            # Test cleanup
            trainer.cleanup()
            print("✓ Cleanup completed")

        else:
            print("✗ Failed to initialize distributed training")

    # Run test
    test_distributed_setup()