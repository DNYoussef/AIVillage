import torch
import torch.nn as nn
import math

class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scale = torch.mean(torch.abs(input))
        return torch.round(torch.clamp(input / scale, -1, 1)).to(torch.int8) * scale

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.quantizer = TernaryQuantizer.apply

    def forward(self, input):
        quantized_weight = self.quantizer(self.weight)
        return nn.functional.linear(input, quantized_weight, self.bias)

def convert_to_bitnet(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear(module.in_features, module.out_features, module.bias is not None))
        else:
            convert_to_bitnet(module)
    return model

def quantize_activations(x):
    scale = 127.0 / torch.max(torch.abs(x))
    return torch.round(torch.clamp(x * scale, -127, 127)) / scale

class BitNetModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = convert_to_bitnet(original_model)

    def forward(self, x):
        for name, module in self.model.named_children():
            if isinstance(module, BitLinear):
                x = quantize_activations(x)
            x = module(x)
        return x

def fine_tune(model, train_loader, val_loader, epochs=5, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy:.4f}")

# Example usage
def main():
    # Assume we have a pre-trained model
    original_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    # Convert to BitNet model
    bitnet_model = BitNetModel(original_model)
    
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
    
    # Fine-tune the model
    fine_tune(bitnet_model, train_loader, val_loader)
    
    # Save the quantized model
    torch.save(bitnet_model.state_dict(), 'bitnet_model.pth')
    
    print("Conversion and fine-tuning complete. Model saved as 'bitnet_model.pth'")

if __name__ == "__main__":
    main()