import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scale = torch.mean(torch.abs(input)).clamp(min=1e-8)
        input_scaled = input / scale
        output = torch.zeros_like(input)
        output[input_scaled > 0.3] = 1
        output[input_scaled < -0.3] = -1
        return output * scale, scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1] = 0
        return grad_input

class CustomQuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('weight_scale', torch.ones(1))

    def forward(self, x):
        quantized_weight, self.weight_scale = TernaryQuantizer.apply(self.weight)
        return F.linear(x, quantized_weight, self.bias)

def quantize_activations(x):
    scale = 127.0 / torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return torch.round(torch.clamp(x * scale, -127, 127)) / scale

class FrozenEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.freeze_weights()

    def freeze_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)

class FrozenAutoencoder(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
        self.freeze_weights()

    def freeze_weights(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.autoencoder(x)

class SleepBlock(nn.Module):
    def __init__(self, in_features, out_features, encoder):
        super().__init__()
        self.chain_block = nn.Sequential(
            CustomQuantizedLinear(in_features, out_features),
            nn.ReLU()
        )
        self.encoder = encoder

    def forward(self, x):
        chain_output = self.chain_block(x)
        encoded_output = self.encoder(x)
        return quantize_activations(chain_output + encoded_output)

class DreamBlock(nn.Module):
    def __init__(self, in_features, out_features, autoencoder):
        super().__init__()
        self.chain_block = nn.Sequential(
            CustomQuantizedLinear(in_features, out_features),
            nn.ReLU()
        )
        self.autoencoder = autoencoder

    def forward(self, x):
        chain_output = self.chain_block(x)
        dream_output = self.autoencoder(x)
        return quantize_activations(chain_output + dream_output)

class SleepNet(nn.Module):
    def __init__(self, input_size, output_size, num_sleep_blocks, model_type='vit-base', freeze_encoder=True):
        super().__init__()
        self.input_layer = CustomQuantizedLinear(input_size, input_size)
        
        if 'vit' in model_type:
            self.pretrained_encoder = AutoModel.from_pretrained(f"google/{model_type}-patch16-224")
        elif 'roberta' in model_type:
            self.pretrained_encoder = AutoModel.from_pretrained(f"roberta-{model_type.split('-')[1]}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if freeze_encoder:
            self.pretrained_encoder = FrozenEncoder(self.pretrained_encoder)
        
        self.sleep_blocks = nn.ModuleList([
            SleepBlock(input_size, input_size, self.pretrained_encoder)
            for _ in range(num_sleep_blocks)
        ])
        self.output_layer = CustomQuantizedLinear(input_size, output_size)

    def forward(self, x):
        x = quantize_activations(self.input_layer(x))
        for sleep_block in self.sleep_blocks:
            x = sleep_block(x)
        return self.output_layer(x)

class DreamNet(nn.Module):
    def __init__(self, input_size, output_size, num_dream_blocks, model_type='mae-base', freeze_autoencoder=True):
        super().__init__()
        self.input_layer = CustomQuantizedLinear(input_size, input_size)
        
        if 'mae' in model_type:
            self.pretrained_autoencoder = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_type}")
        elif 'xlnet' in model_type:
            self.pretrained_autoencoder = AutoModelForCausalLM.from_pretrained(f"xlnet-{model_type.split('-')[1]}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if freeze_autoencoder:
            self.pretrained_autoencoder = FrozenAutoencoder(self.pretrained_autoencoder)
        
        self.dream_blocks = nn.ModuleList([
            DreamBlock(input_size, input_size, self.pretrained_autoencoder)
            for _ in range(num_dream_blocks)
        ])
        self.output_layer = CustomQuantizedLinear(input_size, output_size)

    def forward(self, x):
        x = quantize_activations(self.input_layer(x))
        for dream_block in self.dream_blocks:
            x = dream_block(x)
        return self.output_layer(x)

# Usage example:
# sleepnet_cv = SleepNet(input_size=768, output_size=num_classes, num_sleep_blocks=3, model_type='vit-base')
# dreamnet_cv = DreamNet(input_size=768, output_size=num_classes, num_dream_blocks=3, model_type='mae-base')
# sleepnet_nlp = SleepNet(input_size=768, output_size=num_classes, num_sleep_blocks=3, model_type='roberta-base')
# dreamnet_nlp = DreamNet(input_size=768, output_size=num_classes, num_dream_blocks=3, model_type='xlnet-base')