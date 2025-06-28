import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
from langroid import Task, ChatAgent, ChatAgentConfig

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
    def __init__(self, input_size, output_size, num_sleep_blocks, model_type='vit-base', freeze_encoder=True, pretrained=False):
        """Lightweight SleepNet wrapper.

        When ``pretrained`` is ``False`` (default) a simple ``nn.Identity`` is
        used instead of downloading large vision models so that unit tests can
        run without network access. Set ``pretrained=True`` to restore the
        original behaviour.
        """
        super().__init__()
        self.input_layer = CustomQuantizedLinear(input_size, input_size)

        if pretrained:
            if 'vit' in model_type:
                self.pretrained_encoder = AutoModel.from_pretrained(f"google/{model_type}-patch16-224")
            elif 'roberta' in model_type:
                self.pretrained_encoder = AutoModel.from_pretrained(f"roberta-{model_type.split('-')[1]}")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            if freeze_encoder:
                self.pretrained_encoder = FrozenEncoder(self.pretrained_encoder)
        else:
            self.pretrained_encoder = nn.Identity()
        
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
    def __init__(self, input_size, output_size, num_dream_blocks, model_type='mae-base', freeze_autoencoder=True, pretrained=False):
        """Mirror of ``SleepNet`` for the dream phase."""
        super().__init__()
        self.input_layer = CustomQuantizedLinear(input_size, input_size)

        if pretrained:
            if 'mae' in model_type:
                self.pretrained_autoencoder = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_type}")
            elif 'xlnet' in model_type:
                self.pretrained_autoencoder = AutoModelForCausalLM.from_pretrained(f"xlnet-{model_type.split('-')[1]}")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            if freeze_autoencoder:
                self.pretrained_autoencoder = FrozenAutoencoder(self.pretrained_autoencoder)
        else:
            self.pretrained_autoencoder = nn.Identity()
        
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

class SleepAndDreamTask(Task):
    def __init__(self, agent: ChatAgent, input_size: int, output_size: int, num_sleep_blocks: int, num_dream_blocks: int, *, pretrained: bool = False):
        super().__init__(agent)
        self.sleep_net = SleepNet(input_size, output_size, num_sleep_blocks, pretrained=pretrained)
        self.dream_net = DreamNet(input_size, output_size, num_dream_blocks, pretrained=pretrained)

    async def run(self, input_data: torch.Tensor) -> torch.Tensor:
        sleep_output = self.sleep_net(input_data)
        dream_output = self.dream_net(sleep_output)
        return dream_output

# Usage example
if __name__ == "__main__":
    import asyncio
    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main():
        config = ChatAgentConfig(
            name="SleepAndDreamAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        task = SleepAndDreamTask(agent, input_size=768, output_size=768, num_sleep_blocks=3, num_dream_blocks=3, pretrained=False)
        
        # Example input tensor
        input_data = torch.randn(1, 768)
        
        result = await task.run(input_data)
        print("Sleep and Dream process completed. Output shape:", result.shape)

    asyncio.run(main())
