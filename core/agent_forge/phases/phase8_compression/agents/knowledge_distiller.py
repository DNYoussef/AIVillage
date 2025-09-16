import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import copy
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    distillation_type: str  # 'response', 'feature', 'attention', 'relation'
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student loss
    student_architecture: Optional[Dict[str, Any]] = None
    feature_layers: Optional[List[str]] = None
    num_epochs: int = 100
    learning_rate: float = 0.001
    
@dataclass
class DistillationResult:
    """Results from knowledge distillation."""
    teacher_accuracy: float
    student_accuracy: float
    compression_ratio: float
    knowledge_transfer_efficiency: float
    training_metrics: Dict[str, List[float]]
    final_loss: float
    
class DistillationStrategy(ABC):
    """Abstract base class for distillation strategies."""
    
    @abstractmethod
    def distill(self, teacher_model: nn.Module, 
               student_model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: DistillationConfig) -> DistillationResult:
        pass
        
class ResponseDistillation(DistillationStrategy):
    """Response-based knowledge distillation."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def distill(self, teacher_model: nn.Module, 
               student_model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: DistillationConfig) -> DistillationResult:
        """Perform response-based distillation."""
        try:
            # Move models to device
            teacher_model.to(self.device)
            student_model.to(self.device)
            
            # Set teacher to eval mode
            teacher_model.eval()
            
            # Setup optimizer
            optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
            
            # Training metrics
            train_losses = []
            val_accuracies = []
            
            # Training loop
            for epoch in range(config.num_epochs):
                # Training phase
                student_model.train()
                epoch_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Get teacher predictions
                    with torch.no_grad():
                        teacher_output = teacher_model(data)
                        
                    # Get student predictions
                    student_output = student_model(data)
                    
                    # Compute losses
                    distillation_loss = self._distillation_loss(
                        student_output, teacher_output, config.temperature
                    )
                    
                    student_loss = F.cross_entropy(student_output, target)
                    
                    # Combined loss
                    total_loss = config.alpha * distillation_loss + config.beta * student_loss
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    
                scheduler.step()
                
                # Validation
                val_acc = self._evaluate_model(student_model, val_loader)
                
                train_losses.append(epoch_loss / len(train_loader))
                val_accuracies.append(val_acc)
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss={train_losses[-1]:.4f}, Val Acc={val_acc:.4f}")
                    
            # Calculate final results
            teacher_acc = self._evaluate_model(teacher_model, val_loader)
            student_acc = self._evaluate_model(student_model, val_loader)
            
            # Calculate compression ratio
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in student_model.parameters())
            compression_ratio = teacher_params / student_params
            
            # Knowledge transfer efficiency
            transfer_efficiency = student_acc / teacher_acc
            
            return DistillationResult(
                teacher_accuracy=teacher_acc,
                student_accuracy=student_acc,
                compression_ratio=compression_ratio,
                knowledge_transfer_efficiency=transfer_efficiency,
                training_metrics={'losses': train_losses, 'val_accuracies': val_accuracies},
                final_loss=train_losses[-1]
            )
            
        except Exception as e:
            self.logger.error(f"Response distillation failed: {e}")
            raise
            
    def _distillation_loss(self, student_output: torch.Tensor, 
                          teacher_output: torch.Tensor, 
                          temperature: float) -> torch.Tensor:
        """Compute distillation loss using soft targets."""
        # Apply temperature to soften distributions
        student_soft = F.log_softmax(student_output / temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / temperature, dim=1)
        
        # KL divergence loss
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by temperature squared (as per Hinton et al.)
        return loss * (temperature ** 2)
        
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
class FeatureDistillation(DistillationStrategy):
    """Feature-based knowledge distillation."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def distill(self, teacher_model: nn.Module, 
               student_model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: DistillationConfig) -> DistillationResult:
        """Perform feature-based distillation."""
        try:
            # Move models to device
            teacher_model.to(self.device)
            student_model.to(self.device)
            
            # Set teacher to eval mode
            teacher_model.eval()
            
            # Setup feature extraction hooks
            teacher_features = {}
            student_features = {}
            
            self._register_feature_hooks(teacher_model, teacher_features, config.feature_layers)
            self._register_feature_hooks(student_model, student_features, config.feature_layers)
            
            # Setup optimizer
            optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
            
            # Training metrics
            train_losses = []
            val_accuracies = []
            
            # Training loop
            for epoch in range(config.num_epochs):
                student_model.train()
                epoch_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Clear feature storage
                    teacher_features.clear()
                    student_features.clear()
                    
                    # Forward pass
                    with torch.no_grad():
                        teacher_output = teacher_model(data)
                        
                    student_output = student_model(data)
                    
                    # Compute losses
                    feature_loss = self._feature_matching_loss(teacher_features, student_features)
                    response_loss = self._distillation_loss(student_output, teacher_output, config.temperature)
                    student_loss = F.cross_entropy(student_output, target)
                    
                    # Combined loss
                    total_loss = (config.alpha * (response_loss + feature_loss) + 
                                config.beta * student_loss)
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    
                scheduler.step()
                
                # Validation
                val_acc = self._evaluate_model(student_model, val_loader)
                
                train_losses.append(epoch_loss / len(train_loader))
                val_accuracies.append(val_acc)
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss={train_losses[-1]:.4f}, Val Acc={val_acc:.4f}")
                    
            # Calculate final results
            teacher_acc = self._evaluate_model(teacher_model, val_loader)
            student_acc = self._evaluate_model(student_model, val_loader)
            
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in student_model.parameters())
            compression_ratio = teacher_params / student_params
            
            transfer_efficiency = student_acc / teacher_acc
            
            return DistillationResult(
                teacher_accuracy=teacher_acc,
                student_accuracy=student_acc,
                compression_ratio=compression_ratio,
                knowledge_transfer_efficiency=transfer_efficiency,
                training_metrics={'losses': train_losses, 'val_accuracies': val_accuracies},
                final_loss=train_losses[-1]
            )
            
        except Exception as e:
            self.logger.error(f"Feature distillation failed: {e}")
            raise
            
    def _register_feature_hooks(self, model: nn.Module, feature_dict: Dict, layer_names: List[str]) -> None:
        """Register hooks to extract intermediate features."""
        def hook_fn(name):
            def hook(module, input, output):
                feature_dict[name] = output
            return hook
            
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    module.register_forward_hook(hook_fn(name))
                    
    def _feature_matching_loss(self, teacher_features: Dict, student_features: Dict) -> torch.Tensor:
        """Compute feature matching loss."""
        total_loss = 0.0
        num_features = 0
        
        for name in teacher_features:
            if name in student_features:
                teacher_feat = teacher_features[name]
                student_feat = student_features[name]
                
                # Handle dimension mismatch
                if teacher_feat.shape != student_feat.shape:
                    student_feat = self._adapt_feature_dimensions(student_feat, teacher_feat.shape)
                    
                # MSE loss between features
                loss = F.mse_loss(student_feat, teacher_feat)
                total_loss += loss
                num_features += 1
                
        return total_loss / max(num_features, 1)
        
    def _adapt_feature_dimensions(self, student_feat: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Adapt student feature dimensions to match teacher."""
        # Simple adaptation using adaptive pooling
        if len(target_shape) == 4:  # Conv features
            return F.adaptive_avg_pool2d(student_feat, target_shape[2:])
        elif len(target_shape) == 2:  # Linear features
            return F.adaptive_avg_pool1d(student_feat.unsqueeze(1), target_shape[1]).squeeze(1)
        else:
            return student_feat
            
    def _distillation_loss(self, student_output: torch.Tensor, 
                          teacher_output: torch.Tensor, 
                          temperature: float) -> torch.Tensor:
        """Compute distillation loss."""
        student_soft = F.log_softmax(student_output / temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
        
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
class AttentionDistillation(DistillationStrategy):
    """Attention-based knowledge distillation."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def distill(self, teacher_model: nn.Module, 
               student_model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: DistillationConfig) -> DistillationResult:
        """Perform attention-based distillation."""
        try:
            # Move models to device
            teacher_model.to(self.device)
            student_model.to(self.device)
            
            teacher_model.eval()
            
            # Setup attention extraction
            teacher_attentions = {}
            student_attentions = {}
            
            self._register_attention_hooks(teacher_model, teacher_attentions)
            self._register_attention_hooks(student_model, student_attentions)
            
            # Setup optimizer
            optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
            
            # Training metrics
            train_losses = []
            val_accuracies = []
            
            # Training loop
            for epoch in range(config.num_epochs):
                student_model.train()
                epoch_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Clear attention storage
                    teacher_attentions.clear()
                    student_attentions.clear()
                    
                    # Forward pass
                    with torch.no_grad():
                        teacher_output = teacher_model(data)
                        
                    student_output = student_model(data)
                    
                    # Compute losses
                    attention_loss = self._attention_transfer_loss(teacher_attentions, student_attentions)
                    response_loss = self._distillation_loss(student_output, teacher_output, config.temperature)
                    student_loss = F.cross_entropy(student_output, target)
                    
                    # Combined loss
                    total_loss = (config.alpha * (response_loss + attention_loss) + 
                                config.beta * student_loss)
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    
                scheduler.step()
                
                # Validation
                val_acc = self._evaluate_model(student_model, val_loader)
                
                train_losses.append(epoch_loss / len(train_loader))
                val_accuracies.append(val_acc)
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss={train_losses[-1]:.4f}, Val Acc={val_acc:.4f}")
                    
            # Calculate final results
            teacher_acc = self._evaluate_model(teacher_model, val_loader)
            student_acc = self._evaluate_model(student_model, val_loader)
            
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in student_model.parameters())
            compression_ratio = teacher_params / student_params
            
            transfer_efficiency = student_acc / teacher_acc
            
            return DistillationResult(
                teacher_accuracy=teacher_acc,
                student_accuracy=student_acc,
                compression_ratio=compression_ratio,
                knowledge_transfer_efficiency=transfer_efficiency,
                training_metrics={'losses': train_losses, 'val_accuracies': val_accuracies},
                final_loss=train_losses[-1]
            )
            
        except Exception as e:
            self.logger.error(f"Attention distillation failed: {e}")
            raise
            
    def _register_attention_hooks(self, model: nn.Module, attention_dict: Dict) -> None:
        """Register hooks to extract attention maps."""
        def attention_hook(name):
            def hook(module, input, output):
                if isinstance(module, nn.Conv2d):
                    # Generate attention map from conv features
                    attention_map = torch.mean(output, dim=1, keepdim=True)
                    attention_dict[name] = attention_map
            return hook
            
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(attention_hook(name))
                
    def _attention_transfer_loss(self, teacher_attentions: Dict, student_attentions: Dict) -> torch.Tensor:
        """Compute attention transfer loss."""
        total_loss = 0.0
        num_attentions = 0
        
        for name in teacher_attentions:
            if name in student_attentions:
                teacher_att = teacher_attentions[name]
                student_att = student_attentions[name]
                
                # Normalize attention maps
                teacher_att = F.normalize(teacher_att.flatten(2), p=2, dim=2)
                student_att = F.normalize(student_att.flatten(2), p=2, dim=2)
                
                # Attention transfer loss
                loss = F.mse_loss(student_att, teacher_att)
                total_loss += loss
                num_attentions += 1
                
        return total_loss / max(num_attentions, 1)
        
    def _distillation_loss(self, student_output: torch.Tensor, 
                          teacher_output: torch.Tensor, 
                          temperature: float) -> torch.Tensor:
        """Compute distillation loss."""
        student_soft = F.log_softmax(student_output / temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
        
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
class KnowledgeDistiller:
    """Main knowledge distillation agent."""
    
    def __init__(self, device: torch.device = torch.device('cpu'), logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.strategies = {
            'response': ResponseDistillation(device, logger),
            'feature': FeatureDistillation(device, logger),
            'attention': AttentionDistillation(device, logger)
        }
        
    def create_student_model(self, teacher_model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
        """Create a smaller student model based on teacher architecture."""
        try:
            # Simple compression by reducing layer dimensions
            student_model = copy.deepcopy(teacher_model)
            
            for name, module in student_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Reduce number of output channels
                    new_out_channels = max(1, int(module.out_channels * compression_ratio))
                    new_conv = nn.Conv2d(
                        module.in_channels,
                        new_out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        module.bias is not None
                    )
                    # Copy weights (truncated)
                    with torch.no_grad():
                        new_conv.weight.data = module.weight.data[:new_out_channels].clone()
                        if module.bias is not None:
                            new_conv.bias.data = module.bias.data[:new_out_channels].clone()
                    
                    # Replace module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = dict(student_model.named_modules())[parent_name]
                        setattr(parent, child_name, new_conv)
                    
                elif isinstance(module, nn.Linear):
                    # Reduce number of output features
                    new_out_features = max(1, int(module.out_features * compression_ratio))
                    new_linear = nn.Linear(
                        module.in_features,
                        new_out_features,
                        module.bias is not None
                    )
                    # Copy weights (truncated)
                    with torch.no_grad():
                        new_linear.weight.data = module.weight.data[:new_out_features].clone()
                        if module.bias is not None:
                            new_linear.bias.data = module.bias.data[:new_out_features].clone()
                    
                    # Replace module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = dict(student_model.named_modules())[parent_name]
                        setattr(parent, child_name, new_linear)
                        
            return student_model
            
        except Exception as e:
            self.logger.error(f"Failed to create student model: {e}")
            raise
            
    def distill_knowledge(self, teacher_model: nn.Module,
                         student_model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         config: DistillationConfig) -> DistillationResult:
        """Perform knowledge distillation using specified strategy."""
        try:
            if config.distillation_type not in self.strategies:
                raise ValueError(f"Unknown distillation type: {config.distillation_type}")
                
            strategy = self.strategies[config.distillation_type]
            result = strategy.distill(teacher_model, student_model, train_loader, val_loader, config)
            
            self.logger.info(f"Distillation completed: {result.compression_ratio:.2f}x compression, "
                           f"Transfer efficiency: {result.knowledge_transfer_efficiency:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Knowledge distillation failed: {e}")
            raise
            
    def compare_distillation_methods(self, teacher_model: nn.Module,
                                   train_loader: DataLoader,
                                   val_loader: DataLoader,
                                   compression_ratio: float = 0.5) -> Dict[str, DistillationResult]:
        """Compare different distillation methods."""
        results = {}
        
        methods = ['response', 'feature', 'attention']
        
        for method in methods:
            try:
                self.logger.info(f"Testing {method} distillation")
                
                # Create student model
                student_model = self.create_student_model(teacher_model, compression_ratio)
                
                # Create config
                config = DistillationConfig(
                    distillation_type=method,
                    num_epochs=20,  # Reduced for comparison
                    learning_rate=0.001
                )
                
                # Perform distillation
                result = self.distill_knowledge(
                    copy.deepcopy(teacher_model),
                    student_model,
                    train_loader,
                    val_loader,
                    config
                )
                
                results[method] = result
                
            except Exception as e:
                self.logger.error(f"Failed to test {method} distillation: {e}")
                
        return results
        
    def create_distillation_config(self, distillation_type: str, **kwargs) -> DistillationConfig:
        """Create distillation configuration."""
        return DistillationConfig(
            distillation_type=distillation_type,
            **kwargs
        )
        
    def analyze_knowledge_transfer(self, teacher_model: nn.Module, 
                                 student_model: nn.Module,
                                 test_loader: DataLoader) -> Dict[str, float]:
        """Analyze the quality of knowledge transfer."""
        try:
            teacher_model.eval()
            student_model.eval()
            
            # Calculate various metrics
            teacher_acc = self._evaluate_accuracy(teacher_model, test_loader)
            student_acc = self._evaluate_accuracy(student_model, test_loader)
            
            # Model sizes
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in student_model.parameters())
            
            # Feature similarity
            feature_similarity = self._calculate_feature_similarity(teacher_model, student_model, test_loader)
            
            # Output distribution similarity
            output_similarity = self._calculate_output_similarity(teacher_model, student_model, test_loader)
            
            analysis = {
                'teacher_accuracy': teacher_acc,
                'student_accuracy': student_acc,
                'accuracy_retention': student_acc / teacher_acc,
                'compression_ratio': teacher_params / student_params,
                'parameter_efficiency': student_acc / student_params * 1e6,  # Accuracy per million params
                'feature_similarity': feature_similarity,
                'output_similarity': output_similarity,
                'knowledge_transfer_score': (student_acc / teacher_acc) * np.log(teacher_params / student_params)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Knowledge transfer analysis failed: {e}")
            raise
            
    def _evaluate_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
    def _calculate_feature_similarity(self, teacher_model: nn.Module, 
                                    student_model: nn.Module, 
                                    data_loader: DataLoader) -> float:
        """Calculate feature similarity between teacher and student."""
        teacher_features = []
        student_features = []
        
        def feature_hook(feature_list):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    feature_list.append(output.detach())
            return hook
            
        # Register hooks on final conv/linear layers
        teacher_hooks = []
        student_hooks = []
        
        for name, module in teacher_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and len(list(module.children())) == 0:
                teacher_hooks.append(module.register_forward_hook(feature_hook(teacher_features)))
                break
                
        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and len(list(module.children())) == 0:
                student_hooks.append(module.register_forward_hook(feature_hook(student_features)))
                break
                
        # Compute features
        similarities = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 100:  # Limit for efficiency
                    break
                    
                data = data.to(self.device)
                
                teacher_features.clear()
                student_features.clear()
                
                teacher_model(data)
                student_model(data)
                
                if teacher_features and student_features:
                    # Calculate cosine similarity
                    teacher_feat = teacher_features[0].flatten(1)
                    student_feat = student_features[0].flatten(1)
                    
                    # Handle dimension mismatch
                    min_dim = min(teacher_feat.size(1), student_feat.size(1))
                    teacher_feat = teacher_feat[:, :min_dim]
                    student_feat = student_feat[:, :min_dim]
                    
                    similarity = F.cosine_similarity(teacher_feat, student_feat, dim=1).mean().item()
                    similarities.append(similarity)
                    
        # Clean up hooks
        for hook in teacher_hooks + student_hooks:
            hook.remove()
            
        return np.mean(similarities) if similarities else 0.0
        
    def _calculate_output_similarity(self, teacher_model: nn.Module, 
                                   student_model: nn.Module, 
                                   data_loader: DataLoader) -> float:
        """Calculate output distribution similarity."""
        similarities = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 100:  # Limit for efficiency
                    break
                    
                data = data.to(self.device)
                
                teacher_output = teacher_model(data)
                student_output = student_model(data)
                
                # Convert to probabilities
                teacher_probs = F.softmax(teacher_output, dim=1)
                student_probs = F.softmax(student_output, dim=1)
                
                # KL divergence similarity (lower is better, so we use negative)
                kl_div = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
                similarity = torch.exp(-kl_div).item()  # Convert to similarity score
                similarities.append(similarity)
                
        return np.mean(similarities) if similarities else 0.0
