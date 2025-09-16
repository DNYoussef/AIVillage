"""
Knowledge Distillation Framework
Implements teacher-student distillation techniques for model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student task loss
    distillation_type: str = "response"  # response, feature, attention
    feature_matching_layers: Optional[List[str]] = None
    attention_matching: bool = False
    progressive_distillation: bool = False

class BaseDistiller(ABC):
    """Base class for knowledge distillation algorithms."""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: DistillationConfig):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.feature_hooks = {}
        self.teacher_features = {}
        self.student_features = {}

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    @abstractmethod
    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss."""
        pass

    def distill(self, dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               epochs: int = 10,
               device: str = 'cpu') -> Dict[str, List[float]]:
        """Perform knowledge distillation training."""
        self.teacher_model.to(device)
        self.student_model.to(device)

        history = {'loss': [], 'distill_loss': [], 'task_loss': [], 'accuracy': []}

        for epoch in range(epochs):
            self.student_model.train()
            epoch_losses = {'total': 0, 'distill': 0, 'task': 0}
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                # Teacher forward pass
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)

                # Student forward pass
                student_outputs = self.student_model(inputs)

                # Compute losses
                distill_loss = self.compute_distillation_loss(
                    teacher_outputs, student_outputs, targets
                )
                task_loss = criterion(student_outputs, targets)

                total_loss = (self.config.alpha * distill_loss +
                             self.config.beta * task_loss)

                total_loss.backward()
                optimizer.step()

                # Statistics
                epoch_losses['total'] += total_loss.item()
                epoch_losses['distill'] += distill_loss.item()
                epoch_losses['task'] += task_loss.item()

                _, predicted = torch.max(student_outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}, '
                              f'Loss: {total_loss.item():.6f}')

            # Epoch statistics
            avg_loss = epoch_losses['total'] / len(dataloader)
            avg_distill = epoch_losses['distill'] / len(dataloader)
            avg_task = epoch_losses['task'] / len(dataloader)
            accuracy = 100. * correct / total

            history['loss'].append(avg_loss)
            history['distill_loss'].append(avg_distill)
            history['task_loss'].append(avg_task)
            history['accuracy'].append(accuracy)

            logger.info(f'Epoch {epoch}: Loss={avg_loss:.6f}, '
                       f'Distill={avg_distill:.6f}, Task={avg_task:.6f}, '
                       f'Acc={accuracy:.2f}%')

        return history

    def register_feature_hooks(self, layer_names: List[str]) -> None:
        """Register hooks to capture intermediate features."""
        def get_activation(name):
            def hook(model, input, output):
                if model.training:
                    self.student_features[name] = output
                else:
                    self.teacher_features[name] = output
            return hook

        # Register hooks for both teacher and student
        for name in layer_names:
            # Find layer in both models
            teacher_layer = dict(self.teacher_model.named_modules())[name]
            student_layer = dict(self.student_model.named_modules())[name]

            teacher_layer.register_forward_hook(get_activation(name))
            student_layer.register_forward_hook(get_activation(name))

class ResponseDistiller(BaseDistiller):
    """Response-based knowledge distillation using soft targets."""

    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between teacher and student outputs."""
        # Soften the teacher and student outputs
        teacher_soft = F.softmax(teacher_outputs / self.config.temperature, dim=1)
        student_log_soft = F.log_softmax(student_outputs / self.config.temperature, dim=1)

        # KL divergence loss
        distill_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')

        # Scale by temperature squared (as in original paper)
        distill_loss *= (self.config.temperature ** 2)

        return distill_loss

class FeatureDistiller(BaseDistiller):
    """Feature-based knowledge distillation matching intermediate representations."""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: DistillationConfig):
        super().__init__(teacher_model, student_model, config)

        if not config.feature_matching_layers:
            raise ValueError("Feature matching layers must be specified for FeatureDistiller")

        self.register_feature_hooks(config.feature_matching_layers)
        self.adaptation_layers = nn.ModuleDict()

        # Create adaptation layers to match feature dimensions
        self._create_adaptation_layers()

    def _create_adaptation_layers(self) -> None:
        """Create adaptation layers to match teacher-student feature dimensions."""
        for layer_name in self.config.feature_matching_layers:
            teacher_layer = dict(self.teacher_model.named_modules())[layer_name]
            student_layer = dict(self.student_model.named_modules())[layer_name]

            # Get feature dimensions (simplified - assumes conv or linear layers)
            if hasattr(teacher_layer, 'out_channels'):
                teacher_dim = teacher_layer.out_channels
                student_dim = student_layer.out_channels
            elif hasattr(teacher_layer, 'out_features'):
                teacher_dim = teacher_layer.out_features
                student_dim = student_layer.out_features
            else:
                continue

            # Create adaptation layer if dimensions don't match
            if teacher_dim != student_dim:
                self.adaptation_layers[layer_name] = nn.Conv2d(
                    student_dim, teacher_dim, kernel_size=1
                )

    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute feature matching loss + response distillation."""
        # Response distillation loss
        response_distiller = ResponseDistiller(
            self.teacher_model, self.student_model, self.config
        )
        response_loss = response_distiller.compute_distillation_loss(
            teacher_outputs, student_outputs, targets
        )

        # Feature matching loss
        feature_loss = 0
        for layer_name in self.config.feature_matching_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]

                # Apply adaptation layer if needed
                if layer_name in self.adaptation_layers:
                    student_feat = self.adaptation_layers[layer_name](student_feat)

                # L2 loss between features
                feat_loss = F.mse_loss(student_feat, teacher_feat)
                feature_loss += feat_loss

        # Combine losses
        total_loss = response_loss + 0.5 * feature_loss

        # Clear features for next iteration
        self.teacher_features.clear()
        self.student_features.clear()

        return total_loss

class AttentionDistiller(BaseDistiller):
    """Attention-based knowledge distillation for transformer models."""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: DistillationConfig):
        super().__init__(teacher_model, student_model, config)
        self.teacher_attentions = {}
        self.student_attentions = {}

        # Register attention hooks
        self._register_attention_hooks()

    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        def get_attention_hook(name, is_teacher=True):
            def hook(module, input, output):
                # Assuming output contains attention weights
                if hasattr(module, 'attention_weights'):
                    if is_teacher:
                        self.teacher_attentions[name] = module.attention_weights
                    else:
                        self.student_attentions[name] = module.attention_weights
            return hook

        # Find attention layers (simplified - assumes specific naming)
        for name, module in self.teacher_model.named_modules():
            if 'attention' in name.lower():
                module.register_forward_hook(get_attention_hook(name, True))

        for name, module in self.student_model.named_modules():
            if 'attention' in name.lower():
                module.register_forward_hook(get_attention_hook(name, False))

    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute attention transfer loss."""
        # Response distillation
        response_distiller = ResponseDistiller(
            self.teacher_model, self.student_model, self.config
        )
        response_loss = response_distiller.compute_distillation_loss(
            teacher_outputs, student_outputs, targets
        )

        # Attention transfer loss
        attention_loss = 0
        matched_layers = set(self.teacher_attentions.keys()) & set(self.student_attentions.keys())

        for layer_name in matched_layers:
            teacher_att = self.teacher_attentions[layer_name]
            student_att = self.student_attentions[layer_name]

            # MSE loss between attention maps
            att_loss = F.mse_loss(student_att, teacher_att)
            attention_loss += att_loss

        total_loss = response_loss + 0.1 * attention_loss

        # Clear attention maps
        self.teacher_attentions.clear()
        self.student_attentions.clear()

        return total_loss

class ProgressiveDistiller(BaseDistiller):
    """Progressive knowledge distillation with curriculum learning."""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: DistillationConfig):
        super().__init__(teacher_model, student_model, config)
        self.temperature_schedule = self._create_temperature_schedule()
        self.current_epoch = 0

    def _create_temperature_schedule(self) -> List[float]:
        """Create temperature schedule for progressive distillation."""
        # Start with high temperature, gradually decrease
        start_temp = self.config.temperature * 2
        end_temp = self.config.temperature

        # Exponential decay schedule
        schedule = []
        for epoch in range(100):  # Assuming max 100 epochs
            temp = start_temp * np.exp(-epoch * 0.05)
            temp = max(temp, end_temp)
            schedule.append(temp)

        return schedule

    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute progressive distillation loss."""
        # Get current temperature
        current_temp = self.temperature_schedule[min(self.current_epoch, len(self.temperature_schedule) - 1)]

        # Compute KL loss with current temperature
        teacher_soft = F.softmax(teacher_outputs / current_temp, dim=1)
        student_log_soft = F.log_softmax(student_outputs / current_temp, dim=1)

        distill_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
        distill_loss *= (current_temp ** 2)

        return distill_loss

    def distill(self, dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               epochs: int = 10,
               device: str = 'cpu') -> Dict[str, List[float]]:
        """Progressive distillation training."""
        history = super().distill(dataloader, optimizer, criterion, epochs, device)
        self.current_epoch += epochs
        return history

class MultiTeacherDistiller(BaseDistiller):
    """Multi-teacher knowledge distillation."""

    def __init__(self, teacher_models: List[nn.Module], student_model: nn.Module,
                 config: DistillationConfig, teacher_weights: Optional[List[float]] = None):
        # Use first teacher as primary for parent class
        super().__init__(teacher_models[0], student_model, config)
        self.teacher_models = teacher_models
        self.teacher_weights = teacher_weights or [1.0 / len(teacher_models)] * len(teacher_models)

        # Freeze all teacher models
        for teacher in self.teacher_models:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()

    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-teacher distillation loss."""
        total_loss = 0

        # Note: This is simplified - in practice, you'd need to modify the training loop
        # to pass all teacher outputs
        for i, (teacher, weight) in enumerate(zip(self.teacher_models, self.teacher_weights)):
            # Compute individual teacher loss
            response_distiller = ResponseDistiller(teacher, self.student_model, self.config)
            teacher_loss = response_distiller.compute_distillation_loss(
                teacher_outputs, student_outputs, targets
            )

            total_loss += weight * teacher_loss

        return total_loss

class OnlineDistiller(BaseDistiller):
    """Online knowledge distillation where student and teacher train together."""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: DistillationConfig):
        super().__init__(teacher_model, student_model, config)

        # Unfreeze teacher for online distillation
        for param in self.teacher_model.parameters():
            param.requires_grad = True

    def distill(self, dataloader: torch.utils.data.DataLoader,
               teacher_optimizer: torch.optim.Optimizer,
               student_optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               epochs: int = 10,
               device: str = 'cpu') -> Dict[str, List[float]]:
        """Online distillation training."""
        self.teacher_model.to(device)
        self.student_model.to(device)

        history = {'teacher_loss': [], 'student_loss': [], 'distill_loss': [],
                  'teacher_acc': [], 'student_acc': []}

        for epoch in range(epochs):
            self.teacher_model.train()
            self.student_model.train()

            epoch_losses = {'teacher': 0, 'student': 0, 'distill': 0}
            teacher_correct = student_correct = total = 0

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass for both models
                teacher_outputs = self.teacher_model(inputs)
                student_outputs = self.student_model(inputs)

                # Compute losses
                teacher_task_loss = criterion(teacher_outputs, targets)
                student_task_loss = criterion(student_outputs, targets)

                distill_loss = self.compute_distillation_loss(
                    teacher_outputs, student_outputs, targets
                )

                # Update teacher
                teacher_optimizer.zero_grad()
                teacher_total_loss = teacher_task_loss + 0.1 * distill_loss
                teacher_total_loss.backward(retain_graph=True)
                teacher_optimizer.step()

                # Update student
                student_optimizer.zero_grad()
                student_total_loss = student_task_loss + self.config.alpha * distill_loss
                student_total_loss.backward()
                student_optimizer.step()

                # Statistics
                epoch_losses['teacher'] += teacher_total_loss.item()
                epoch_losses['student'] += student_total_loss.item()
                epoch_losses['distill'] += distill_loss.item()

                _, teacher_pred = torch.max(teacher_outputs.data, 1)
                _, student_pred = torch.max(student_outputs.data, 1)
                total += targets.size(0)
                teacher_correct += (teacher_pred == targets).sum().item()
                student_correct += (student_pred == targets).sum().item()

            # Epoch statistics
            teacher_acc = 100. * teacher_correct / total
            student_acc = 100. * student_correct / total

            history['teacher_loss'].append(epoch_losses['teacher'] / len(dataloader))
            history['student_loss'].append(epoch_losses['student'] / len(dataloader))
            history['distill_loss'].append(epoch_losses['distill'] / len(dataloader))
            history['teacher_acc'].append(teacher_acc)
            history['student_acc'].append(student_acc)

            logger.info(f'Epoch {epoch}: Teacher Acc={teacher_acc:.2f}%, '
                       f'Student Acc={student_acc:.2f}%')

        return history

    def compute_distillation_loss(self, teacher_outputs: torch.Tensor,
                                student_outputs: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """Bidirectional distillation loss."""
        # Student learns from teacher
        teacher_to_student = ResponseDistiller(
            self.teacher_model, self.student_model, self.config
        ).compute_distillation_loss(teacher_outputs, student_outputs, targets)

        # Teacher learns from student (reverse distillation)
        student_to_teacher = ResponseDistiller(
            self.student_model, self.teacher_model, self.config
        ).compute_distillation_loss(student_outputs, teacher_outputs, targets)

        return teacher_to_student + 0.1 * student_to_teacher

class DistillationOrchestrator:
    """Orchestrates different knowledge distillation techniques."""

    def __init__(self):
        self.distillers = {
            'response': ResponseDistiller,
            'feature': FeatureDistiller,
            'attention': AttentionDistiller,
            'progressive': ProgressiveDistiller,
            'multi_teacher': MultiTeacherDistiller,
            'online': OnlineDistiller
        }

    def create_distiller(self, distillation_type: str,
                        teacher_model: nn.Module,
                        student_model: nn.Module,
                        config: DistillationConfig,
                        **kwargs) -> BaseDistiller:
        """Create distiller instance."""
        if distillation_type not in self.distillers:
            raise ValueError(f"Unknown distillation type: {distillation_type}")

        distiller_class = self.distillers[distillation_type]

        if distillation_type == 'multi_teacher':
            return distiller_class(kwargs['teacher_models'], student_model, config)
        else:
            return distiller_class(teacher_model, student_model, config)

    def compare_distillation_methods(self, teacher_model: nn.Module,
                                   student_model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   methods: List[str],
                                   config: DistillationConfig) -> Dict[str, Dict]:
        """Compare different distillation methods."""
        results = {}

        for method in methods:
            logger.info(f"Testing {method} distillation...")

            try:
                # Create student copy for each method
                student_copy = torch.nn.utils.prune.identity(student_model)

                # Create distiller
                distiller = self.create_distiller(method, teacher_model, student_copy, config)

                # Simple training setup
                optimizer = torch.optim.SGD(student_copy.parameters(), lr=0.01)
                criterion = nn.CrossEntropyLoss()

                # Train for a few epochs
                history = distiller.distill(dataloader, optimizer, criterion, epochs=3)

                results[method] = {
                    'final_accuracy': history['accuracy'][-1] if 'accuracy' in history else 0,
                    'final_loss': history['loss'][-1] if 'loss' in history else 0,
                    'history': history
                }

            except Exception as e:
                logger.error(f"Failed distillation with {method}: {e}")
                results[method] = {'error': str(e)}

        return results

# Utility functions
def calculate_model_similarity(teacher_model: nn.Module,
                             student_model: nn.Module,
                             dataloader: torch.utils.data.DataLoader) -> float:
    """Calculate output similarity between teacher and student."""
    teacher_model.eval()
    student_model.eval()

    similarities = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)

            # Cosine similarity between outputs
            teacher_flat = teacher_outputs.view(teacher_outputs.size(0), -1)
            student_flat = student_outputs.view(student_outputs.size(0), -1)

            similarity = F.cosine_similarity(teacher_flat, student_flat, dim=1)
            similarities.extend(similarity.cpu().numpy())

    return np.mean(similarities)

# Factory function
def create_distiller(teacher_model: nn.Module,
                    student_model: nn.Module,
                    distillation_type: str = "response",
                    temperature: float = 4.0,
                    alpha: float = 0.7) -> BaseDistiller:
    """Factory function to create distillers."""
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        distillation_type=distillation_type
    )

    orchestrator = DistillationOrchestrator()
    return orchestrator.create_distiller(distillation_type, teacher_model, student_model, config)

if __name__ == "__main__":
    # Example usage
    import torchvision.models as models

    # Create teacher and student models
    teacher = models.resnet50(pretrained=True)
    student = models.resnet18(pretrained=False)

    # Create sample data loader
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 1000, (100,))
    )
    dataloader = DataLoader(dummy_data, batch_size=32)

    # Test different distillation methods
    orchestrator = DistillationOrchestrator()
    config = DistillationConfig(temperature=4.0, alpha=0.7)

    methods_to_test = ['response', 'progressive']

    results = orchestrator.compare_distillation_methods(
        teacher, student, dataloader, methods_to_test, config
    )

    for method, stats in results.items():
        if 'error' not in stats:
            print(f"{method}: Final accuracy {stats['final_accuracy']:.2f}%")
        else:
            print(f"{method}: Error - {stats['error']}")