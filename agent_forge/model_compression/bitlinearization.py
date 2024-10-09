import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from langroid import Task, ChatAgent, ChatAgentConfig

class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scale = torch.mean(torch.abs(input)).clamp(min=1e-8)
        return torch.round(torch.clamp(input / scale, -1, 1)).to(torch.int8), scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('quantized_weight', torch.zeros_like(self.weight, dtype=torch.int8))

    def quantize_weight(self):
        self.quantized_weight, self.weight_scale = TernaryQuantizer.apply(self.weight)

    def forward(self, x):
        x_norm = F.layer_norm(x, x.shape[-1:])
        x_quant = x_norm + (quantize_activations(x_norm) - x_norm).detach()
        w_quant = self.quantized_weight.float() * self.weight_scale
        return F.linear(x_quant, w_quant, self.bias)

def quantize_activations(x):
    scale = 127.0 / torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return torch.round(torch.clamp(x * scale, -127, 127)) / scale

def convert_to_bitnet(model: nn.Module) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            bit_linear = BitLinear(module.in_features, module.out_features, module.bias is not None)
            bit_linear.weight.data = module.weight.data
            if module.bias is not None:
                bit_linear.bias.data = module.bias.data
            bit_linear.quantize_weight()
            setattr(model, name, bit_linear)
        else:
            convert_to_bitnet(module)
    return model

class BitNetModel(nn.Module):
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.model = convert_to_bitnet(original_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def calculate_model_size(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        if hasattr(param, 'quantized_weight'):
            param_size += param.quantized_weight.nelement() * 1.58 / 8  # 1.58 bits per weight
            param_size += param.weight_scale.nelement() * 4  # 32-bit float for scale
        else:
            param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)  # Size in MB

class BitlinearizationTask(Task):
    def __init__(self, agent: ChatAgent, model: nn.Module):
        super().__init__(agent)
        self.model = model

    async def fine_tune(self, train_loader: torch.utils.data.DataLoader, 
                        val_loader: torch.utils.data.DataLoader, epochs: int = 5, 
                        lr: float = 1e-4) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Re-quantize weights after each update
                for module in self.model.modules():
                    if isinstance(module, BitLinear):
                        module.quantize_weight()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = correct / total
            history['val_acc'].append(accuracy)
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = self.model.state_dict()
            
            await self.agent.llm_response(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            scheduler.step()
        
        return {'best_model': best_model, 'best_acc': best_acc, 'history': history}

    async def run(self, train_loader: torch.utils.data.DataLoader, 
                  val_loader: torch.utils.data.DataLoader, epochs: int = 5, 
                  lr: float = 1e-4) -> Dict[str, Any]:
        bitnet_model = BitNetModel(self.model)
        
        original_size = calculate_model_size(self.model)
        bitnet_size = calculate_model_size(bitnet_model)
        
        await self.agent.llm_response(f"Original model size: {original_size:.2f} MB")
        await self.agent.llm_response(f"BitNet model size: {bitnet_size:.2f} MB")
        
        results = await self.fine_tune(bitnet_model, train_loader, val_loader, epochs, lr)
        
        await self.agent.llm_response(f"Best validation accuracy: {results['best_acc']:.4f}")
        
        return results

# Example usage
if __name__ == "__main__":
    import asyncio
    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main():
        config = ChatAgentConfig(
            name="BitlinearizationAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        
        # Assuming you have a pretrained model and data loaders
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Create dummy data loaders (replace with your actual data)
        train_loader = torch.utils.data.DataLoader(
            torch.randn(1000, 3, 224, 224),
            batch_size=32,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.randn(200, 3, 224, 224),
            batch_size=32,
            shuffle=False
        )
        
        task = BitlinearizationTask(agent, pretrained_model)
        results = await task.run(train_loader, val_loader)
        
        print(f"Best validation accuracy: {results['best_acc']:.4f}")

    asyncio.run(main())
