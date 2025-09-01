"""
Dedicated Cognate Pretraining Service

This service handles the complete pretraining pipeline for the 25M parameter Cognate model,
including GrokFast optimization, ACT mechanisms, and LTM cross-attention.
Completely separate from general agent training workflows.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CognateConfig:
    """Configuration for Cognate 25M parameter model"""

    # Model Architecture
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1

    # ACT Parameters
    act_threshold: float = 0.99
    act_max_steps: int = 10
    act_penalty: float = 0.01

    # LTM Parameters
    ltm_memory_size: int = 1024
    ltm_heads: int = 4
    ltm_layers: int = 3

    # Training Parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_steps: int = 100000

    # GrokFast Parameters
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0

    # Checkpointing
    checkpoint_every: int = 1000
    eval_every: int = 500


@dataclass
class CognateTrainingState:
    """Training state for Cognate pretraining"""

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    act_usage: float = 0.0
    ltm_attention_weights: List[float] = None
    grokfast_momentum: Dict[str, torch.Tensor] = None
    learning_rate: float = 0.0
    throughput: float = 0.0
    phase: str = "initialization"


class CognateAttention(nn.Module):
    """Multi-head attention with ACT and LTM support"""

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = self.d_head**-0.5

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, ltm_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Standard attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # LTM cross-attention if memory is provided
        if ltm_memory is not None:
            ltm_k = self.k_proj(ltm_memory).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            ltm_v = self.v_proj(ltm_memory).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

            ltm_attn = torch.matmul(q, ltm_k.transpose(-2, -1)) * self.scale
            attn_weights = torch.cat([ltm_attn, attn_weights], dim=-1)
            v = torch.cat([ltm_v, v], dim=-2)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)

        return out, attn_weights


class CognateACTLayer(nn.Module):
    """Adaptive Computation Time layer for dynamic computation"""

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.config = config
        self.halting_predictor = nn.Linear(config.d_model, 1)
        self.threshold = config.act_threshold
        self.max_steps = config.act_max_steps
        self.penalty = config.act_penalty

    def forward(self, x: torch.Tensor, layer_fn) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape

        # Initialize states
        state = x
        halting_prob = torch.zeros(batch_size, seq_len, 1, device=x.device)
        remainders = torch.ones(batch_size, seq_len, 1, device=x.device)
        n_updates = torch.zeros(batch_size, seq_len, 1, device=x.device)
        output = torch.zeros_like(x)

        for step in range(self.max_steps):
            # Compute halting probabilities
            p = torch.sigmoid(self.halting_predictor(state))

            # Update masks
            still_running = (halting_prob < self.threshold).float()
            new_halted = (halting_prob + p * still_running >= self.threshold).float()

            # For newly halted, use remainder
            p = torch.where(new_halted > 0, remainders, p * still_running)

            # Apply layer computation
            layer_output = layer_fn(state)

            # Accumulate output
            output += p * layer_output

            # Update states
            halting_prob += p
            remainders -= p
            n_updates += still_running
            state = layer_output

            # Check if all sequences have halted
            if (halting_prob >= self.threshold).all():
                break

        # Compute ponder cost (penalty for computation steps)
        ponder_cost = (n_updates + remainders).mean()

        act_info = {"ponder_cost": ponder_cost, "avg_steps": n_updates.mean(), "halting_prob": halting_prob.mean()}

        return output, act_info


class CognateTransformerLayer(nn.Module):
    """Cognate transformer layer with ACT and LTM"""

    def __init__(self, config: CognateConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention = CognateAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # ACT for dynamic computation
        self.act_layer = CognateACTLayer(config)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, ltm_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        def attention_block(x_input):
            attn_out, attn_weights = self.attention(self.norm1(x_input), mask, ltm_memory)
            return x_input + attn_out

        def ff_block(x_input):
            return x_input + self.feed_forward(self.norm2(x_input))

        # Apply ACT to attention block
        x, act_info_attn = self.act_layer(x, attention_block)

        # Apply ACT to feed-forward block
        x, act_info_ff = self.act_layer(x, ff_block)

        layer_info = {"attention_act": act_info_attn, "feedforward_act": act_info_ff, "layer_idx": self.layer_idx}

        return x, layer_info


class CognateLTMMemory(nn.Module):
    """Long-Term Memory module for cross-attention"""

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.config = config
        self.memory = nn.Parameter(torch.randn(config.ltm_memory_size, config.d_model))
        self.memory_proj = nn.Linear(config.d_model, config.d_model)
        self.gate = nn.Linear(config.d_model * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape

        # Project memory
        memory = self.memory_proj(self.memory).unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention between input and memory
        q = x.mean(dim=1, keepdim=True)  # Global query
        k = memory

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        attn_weights = F.softmax(attn_weights, dim=-1)

        memory_context = torch.matmul(attn_weights, memory)

        # Gating mechanism
        gate_input = torch.cat([x.mean(dim=1, keepdim=True), memory_context], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        # Update memory based on input
        updated_memory = gate * memory_context + (1 - gate) * x.mean(dim=1, keepdim=True)

        return memory, updated_memory.squeeze(1)


class CognateModel(nn.Module):
    """25M Parameter Cognate Model with ACT and LTM"""

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([CognateTransformerLayer(config, i) for i in range(config.n_layers)])

        # LTM Memory
        self.ltm_memory = CognateLTMMemory(config)

        # Output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # LTM Memory
        ltm_memory, updated_memory = self.ltm_memory(x)

        # Transformer layers
        act_info = []
        for layer in self.layers:
            x, layer_info = layer(x, attention_mask, ltm_memory)
            act_info.append(layer_info)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return {"logits": logits, "act_info": act_info, "ltm_memory": ltm_memory, "updated_memory": updated_memory}


class CognateDataset(Dataset):
    """Dataset for Cognate pretraining with GSM8K, HotpotQA, SVAMP"""

    def __init__(self, dataset_type: str, tokenizer, max_length: int = 2048):
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset samples based on type"""
        # This is a placeholder - in practice, load actual datasets
        samples = []

        if self.dataset_type == "gsm8k":
            # Mathematical reasoning problems
            for i in range(1000):  # Placeholder
                samples.append(
                    {
                        "text": "Q: A train travels 60 miles per hour for 2 hours. How far does it travel? A: 60 * 2 = 120 miles.",
                        "type": "math",
                    }
                )
        elif self.dataset_type == "hotpotqa":
            # Multi-hop reasoning
            for i in range(1000):  # Placeholder
                samples.append(
                    {
                        "text": "Q: What is the capital of the country where the Eiffel Tower is located? A: The Eiffel Tower is in France, and the capital of France is Paris.",
                        "type": "reasoning",
                    }
                )
        elif self.dataset_type == "svamp":
            # Math word problems
            for i in range(1000):  # Placeholder
                samples.append(
                    {
                        "text": "Q: If John has 5 apples and gives 2 to Mary, how many does he have left? A: 5 - 2 = 3 apples.",
                        "type": "arithmetic",
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize text (placeholder tokenization)
        tokens = [ord(c) % 1000 for c in sample["text"][: self.max_length]]  # Simple placeholder
        tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad

        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long),
            "attention_mask": torch.ones(len(tokens) - 1, dtype=torch.long),
        }


class GrokFastOptimizer:
    """GrokFast optimization for accelerated grokking in Cognate models"""

    def __init__(self, optimizer: torch.optim.Optimizer, alpha: float = 0.98, lamb: float = 2.0):
        self.optimizer = optimizer
        self.alpha = alpha
        self.lamb = lamb
        self.momentum_buffer = {}

    def step(self):
        """Apply GrokFast optimization step"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.momentum_buffer.get(p, {})

                if "momentum" not in param_state:
                    param_state["momentum"] = torch.zeros_like(p.grad)

                momentum = param_state["momentum"]

                # Update momentum with exponential moving average
                momentum.mul_(self.alpha).add_(p.grad, alpha=1 - self.alpha)

                # Apply GrokFast update
                p.grad.add_(momentum, alpha=self.lamb)

                self.momentum_buffer[p] = param_state

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class CognatePretrainingService:
    """Dedicated service for Cognate model pretraining"""

    def __init__(self):
        self.config = CognateConfig()
        self.model = None
        self.optimizer = None
        self.grokfast_optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.state = CognateTrainingState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.websocket_clients = set()

        # Checkpointing
        self.checkpoint_dir = Path("checkpoints/cognate")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CognatePretrainingService with device: {self.device}")

    async def initialize_model(self):
        """Initialize Cognate model and training components"""
        try:
            self.state.phase = "model_initialization"
            logger.info("Initializing 25M parameter Cognate model...")

            # Create model
            self.model = CognateModel(self.config).to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Cognate model initialized with {total_params:,} parameters")

            # Initialize optimizer
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )

            # Initialize GrokFast optimizer
            self.grokfast_optimizer = GrokFastOptimizer(
                self.optimizer, alpha=self.config.grokfast_alpha, lamb=self.config.grokfast_lambda
            )

            # Initialize scheduler
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.config.warmup_steps
            )

            # Initialize datasets
            train_dataset = CognateDataset("gsm8k", None, self.config.max_seq_len)
            val_dataset = CognateDataset("hotpotqa", None, self.config.max_seq_len)

            self.train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2
            )

            self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)

            self.state.phase = "ready"
            await self._emit_websocket_event("model_initialized", {"total_params": total_params})
            logger.info("Cognate model initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize Cognate model: {e}")
            self.state.phase = "error"
            raise

    async def start_pretraining(self):
        """Start Cognate pretraining pipeline"""
        try:
            if self.model is None:
                await self.initialize_model()

            self.state.phase = "pretraining"
            logger.info("Starting Cognate pretraining...")

            self.model.train()
            start_time = time.time()
            tokens_processed = 0

            for step in range(self.config.max_steps):
                self.state.step = step

                # Training step
                batch = next(iter(self.train_loader))
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                act_info = outputs["act_info"]

                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # Add ACT penalty
                act_penalty = sum(
                    info["attention_act"]["ponder_cost"] + info["feedforward_act"]["ponder_cost"] for info in act_info
                ) / len(act_info)
                total_loss = loss + self.config.act_penalty * act_penalty

                # Backward pass
                self.grokfast_optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimization step
                self.grokfast_optimizer.step()
                self.scheduler.step()

                # Update metrics
                self.state.loss = total_loss.item()
                self.state.perplexity = torch.exp(loss).item()
                self.state.act_usage = act_penalty.item()
                self.state.learning_rate = self.scheduler.get_last_lr()[0]

                tokens_processed += input_ids.numel()
                elapsed_time = time.time() - start_time
                self.state.throughput = tokens_processed / elapsed_time if elapsed_time > 0 else 0

                # Logging and checkpointing
                if step % 100 == 0:
                    logger.info(
                        f"Step {step}: loss={self.state.loss:.4f}, ppl={self.state.perplexity:.2f}, "
                        f"act_usage={self.state.act_usage:.4f}, lr={self.state.learning_rate:.2e}"
                    )
                    await self._emit_websocket_event("training_step", asdict(self.state))

                if step % self.config.checkpoint_every == 0 and step > 0:
                    await self._save_checkpoint(step)

                if step % self.config.eval_every == 0 and step > 0:
                    await self._evaluate()

            self.state.phase = "completed"
            logger.info("Cognate pretraining completed successfully")
            await self._emit_websocket_event("training_completed", {"final_step": step})

        except Exception as e:
            logger.error(f"Pretraining failed: {e}")
            self.state.phase = "error"
            await self._emit_websocket_event("training_error", {"error": str(e)})
            raise

    async def _evaluate(self):
        """Evaluate Cognate model on validation set"""
        try:
            self.model.eval()
            total_loss = 0
            total_act_usage = 0
            num_batches = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["logits"]
                    act_info = outputs["act_info"]

                    # Compute loss
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    act_penalty = sum(
                        info["attention_act"]["ponder_cost"] + info["feedforward_act"]["ponder_cost"]
                        for info in act_info
                    ) / len(act_info)

                    total_loss += loss.item()
                    total_act_usage += act_penalty.item()
                    num_batches += 1

                    if num_batches >= 10:  # Limit validation batches
                        break

            avg_loss = total_loss / num_batches
            avg_act_usage = total_act_usage / num_batches
            val_perplexity = math.exp(avg_loss)

            logger.info(f"Validation: loss={avg_loss:.4f}, ppl={val_perplexity:.2f}, act_usage={avg_act_usage:.4f}")
            await self._emit_websocket_event(
                "validation_results", {"loss": avg_loss, "perplexity": val_perplexity, "act_usage": avg_act_usage}
            )

            self.model.train()

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    async def _save_checkpoint(self, step: int):
        """Save Cognate model checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / f"cognate_step_{step}.pt"

            checkpoint = {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": asdict(self.config),
                "training_state": asdict(self.state),
                "grokfast_momentum": self.grokfast_optimizer.momentum_buffer,
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            await self._emit_websocket_event("checkpoint_saved", {"path": str(checkpoint_path), "step": step})

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def load_checkpoint(self, checkpoint_path: str):
        """Load Cognate model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            if self.model is None:
                await self.initialize_model()

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore training state
            self.state.step = checkpoint["step"]
            if "grokfast_momentum" in checkpoint:
                self.grokfast_optimizer.momentum_buffer = checkpoint["grokfast_momentum"]

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            await self._emit_websocket_event("checkpoint_loaded", {"path": checkpoint_path})

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    async def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        metrics = {
            "state": asdict(self.state),
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "device": str(self.device),
            "memory_usage": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        }
        return metrics

    async def _emit_websocket_event(self, event_type: str, data: Dict[str, Any]):
        """Emit WebSocket event to connected clients"""
        if not self.websocket_clients:
            return

        event = {"type": event_type, "timestamp": time.time(), "service": "cognate_pretraining", "data": data}

        message = json.dumps(event)
        disconnected_clients = set()

        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)

        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients

    async def add_websocket_client(self, websocket):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.websocket_clients)}")

    async def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        self.websocket_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")


# Global service instance
cognate_service = CognatePretrainingService()


async def main():
    """Main function for standalone testing"""
    service = CognatePretrainingService()
    await service.initialize_model()
    await service.start_pretraining()


if __name__ == "__main__":
    asyncio.run(main())
