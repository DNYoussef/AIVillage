"""
LLM Driver Abstraction for Local Models

Supports pluggable local models (7B-14B) like Llama for repair proposals.
Provides unified interface for different model backends.
"""

from typing import Dict, List, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import re
import asyncio
import logging
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime


class ModelBackend(Enum):
    """Supported model backends"""
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class ModelConfig:
    """Configuration for LLM model"""
    model_name: str
    backend: ModelBackend
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\n", "###"])

    # Model-specific parameters
    context_length: int = 4096
    batch_size: int = 1
    gpu_layers: int = -1  # -1 for auto

    # Performance settings
    timeout_seconds: int = 30
    retry_attempts: int = 3

    # Rate limiting
    requests_per_minute: int = 60
    max_concurrent_requests: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "backend": self.backend.value,
            "model_path": self.model_path,
            "api_endpoint": self.api_endpoint,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "context_length": self.context_length,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class GenerationRequest:
    """Request for LLM generation"""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

    def merge_with_config(self, config: ModelConfig) -> Dict[str, Any]:
        """Merge request parameters with model config"""
        return {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens or config.max_tokens,
            "temperature": self.temperature or config.temperature,
            "top_p": config.top_p,
            "stop_sequences": self.stop_sequences or config.stop_sequences,
            "stream": self.stream
        }


@dataclass
class GenerationResponse:
    """Response from LLM generation"""
    text: str
    finish_reason: str
    usage: Dict[str, int]
    model: str
    latency_ms: float

    # Confidence and quality metrics
    confidence_score: Optional[float] = None
    quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "confidence_score": self.confidence_score,
            "quality_score": self.quality_score
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.backend.value}")

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt"""
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate text with streaming"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available and functional"""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        pass


class OllamaBackend(LLMBackend):
    """Ollama backend for local models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.api_endpoint or "http://localhost:11434"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using Ollama API"""
        import aiohttp

        start_time = time.time()

        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "stop": request.stop_sequences or self.config.stop_sequences
            }
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            latency_ms = (time.time() - start_time) * 1000

            return GenerationResponse(
                text=result.get("response", ""),
                finish_reason=result.get("done_reason", "completed"),
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                model=self.config.model_name,
                latency_ms=latency_ms
            )

        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate with streaming using Ollama"""
        import aiohttp

        payload = {
            "model": self.config.model_name,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "num_predict": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": self.config.top_p
            }
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    yield chunk['response']
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            self.logger.error(f"Ollama streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        model_names = [m["name"] for m in models.get("models", [])]
                        return self.config.model_name in model_names
            return False
        except:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/show",
                    json={"name": self.config.model_name}
                ) as response:
                    if response.status == 200:
                        return await response.json()
            return {}
        except:
            return {}


class LMStudioBackend(LLMBackend):
    """LM Studio backend for local models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.api_endpoint or "http://localhost:1234"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using LM Studio OpenAI-compatible API"""
        import aiohttp

        start_time = time.time()

        # LM Studio uses OpenAI-compatible format
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": self.config.top_p,
            "stop": request.stop_sequences or self.config.stop_sequences,
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Extract response from OpenAI format
            choice = result["choices"][0]
            content = choice["message"]["content"]

            return GenerationResponse(
                text=content,
                finish_reason=choice.get("finish_reason", "completed"),
                usage=result.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }),
                model=self.config.model_name,
                latency_ms=latency_ms
            )

        except Exception as e:
            self.logger.error(f"LM Studio generation failed: {e}")
            raise

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate with streaming using LM Studio"""
        import aiohttp

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if "choices" in chunk and chunk["choices"]:
                                        delta = chunk["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    continue

        except Exception as e:
            self.logger.error(f"LM Studio streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if LM Studio is running and has models loaded"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        models = await response.json()
                        model_ids = [m["id"] for m in models.get("data", [])]
                        return len(model_ids) > 0  # Any model loaded
            return False
        except:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get LM Studio model information"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        return await response.json()
            return {}
        except:
            return {}


class HuggingFaceBackend(LLMBackend):
    """Hugging Face Transformers backend"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    async def _load_model(self):
        """Load model and tokenizer"""
        if self.model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path or self.config.model_name
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path or self.config.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            except ImportError:
                raise RuntimeError("transformers library not available")
            except Exception as e:
                self.logger.error(f"Failed to load HuggingFace model: {e}")
                raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using HuggingFace model"""
        await self._load_model()

        start_time = time.time()

        # Prepare prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            full_prompt = request.prompt

        # Tokenize
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)

        # Generate
        max_new_tokens = request.max_tokens or self.config.max_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature or self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True
        )

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResponse(
            text=generated_text,
            finish_reason="completed",
            usage={
                "prompt_tokens": inputs.shape[-1],
                "completion_tokens": outputs.shape[-1] - inputs.shape[-1],
                "total_tokens": outputs.shape[-1]
            },
            model=self.config.model_name,
            latency_ms=latency_ms
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Streaming not implemented for HuggingFace backend"""
        response = await self.generate(request)
        yield response.text

    async def is_available(self) -> bool:
        """Check if HuggingFace backend is available"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace model information"""
        await self._load_model()
        return {
            "model_name": self.config.model_name,
            "model_type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown"
        }


class LLMDriver:
    """Main driver class for LLM operations"""

    def __init__(self, config: ModelConfig):
        """
        Initialize LLM driver with configuration

        Args:
            config: Model configuration
        """
        self.config = config
        self.backend = self._create_backend()
        self.logger = logging.getLogger(__name__)

        # Rate limiting and auditing
        self._request_times = []
        self._concurrent_requests = 0
        self._audit_log = []

    def _create_backend(self) -> LLMBackend:
        """Create appropriate backend based on config"""
        if self.config.backend == ModelBackend.OLLAMA:
            return OllamaBackend(self.config)
        elif self.config.backend == ModelBackend.LMSTUDIO:
            return LMStudioBackend(self.config)
        elif self.config.backend == ModelBackend.HUGGINGFACE:
            return HuggingFaceBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()

        # Clean old request times
        cutoff = now - 60  # 1 minute ago
        self._request_times = [t for t in self._request_times if t > cutoff]

        # Check requests per minute
        if len(self._request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                self.logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Check concurrent requests
        if self._concurrent_requests >= self.config.max_concurrent_requests:
            raise RuntimeError(f"Max concurrent requests ({self.config.max_concurrent_requests}) exceeded")

        # Record this request
        self._request_times.append(now)

    def _log_request(self, prompt: str, system_prompt: Optional[str], response: GenerationResponse):
        """Log request for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model_name,
            "prompt_length": len(prompt),
            "system_prompt_length": len(system_prompt) if system_prompt else 0,
            "response_length": len(response.text),
            "usage": response.usage,
            "latency_ms": response.latency_ms,
            "finish_reason": response.finish_reason
        }

        self._audit_log.append(log_entry)

        # Keep only recent entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> GenerationResponse:
        """
        Generate text from prompt with rate limiting and audit logging

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generation response
        """
        # Check rate limits
        await self._check_rate_limit()

        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )

        self._concurrent_requests += 1
        try:
            for attempt in range(self.config.retry_attempts):
                try:
                    response = await self.backend.generate(request)

                    # Log for audit trail
                    self._log_request(prompt, system_prompt, response)

                    return response
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        self.logger.error(f"Generation failed after {self.config.retry_attempts} attempts: {e}")
                        raise
                    else:
                        self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}, retrying...")
                        await asyncio.sleep(1)
        finally:
            self._concurrent_requests -= 1

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        """
        Generate text with streaming

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks
        """
        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            stream=True,
            **kwargs
        )

        async for chunk in self.backend.generate_stream(request):
            yield chunk

    async def is_ready(self) -> bool:
        """Check if the LLM driver is ready for generation"""
        return await self.backend.is_available()

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = await self.backend.get_model_info()
        info["config"] = self.config.to_dict()
        return info

    def parse_confidence_from_response(self, response_text: str) -> Optional[float]:
        """
        Parse confidence score from model response

        Args:
            response_text: Generated text

        Returns:
            Confidence score if found
        """
        # Look for confidence patterns in the response
        confidence_patterns = [
            r'confidence["\s]*:?\s*([0-9.]+)',
            r'confidence_score["\s]*:?\s*([0-9.]+)',
            r'"confidence"\s*:\s*([0-9.]+)',
            r'I am (\d+(?:\.\d+)?)% confident',
            r'confidence level:?\s*([0-9.]+)'
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    confidence = float(match.group(1))
                    # Normalize to 0-1 range if it looks like a percentage
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    return max(0.0, min(1.0, confidence))
                except ValueError:
                    continue

        return None

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log of recent requests"""
        return self._audit_log.copy()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self._audit_log:
            return {"message": "No usage data available"}

        recent_entries = self._audit_log[-100:]  # Last 100 requests

        total_requests = len(recent_entries)
        total_tokens = sum(entry["usage"].get("total_tokens", 0) for entry in recent_entries)
        avg_latency = sum(entry["latency_ms"] for entry in recent_entries) / total_requests

        return {
            "total_requests": total_requests,
            "total_tokens_used": total_tokens,
            "average_latency_ms": avg_latency,
            "requests_per_minute_limit": self.config.requests_per_minute,
            "current_requests_in_window": len(self._request_times),
            "concurrent_requests": self._concurrent_requests
        }

    @classmethod
    def create_default_config(cls, model_name: str, backend: ModelBackend = ModelBackend.OLLAMA) -> ModelConfig:
        """
        Create default configuration for common models

        Args:
            model_name: Name of the model
            backend: Backend to use

        Returns:
            Default model configuration
        """
        return ModelConfig(
            model_name=model_name,
            backend=backend,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            context_length=4096,
            timeout_seconds=30
        )

    @classmethod
    def create_llama_config(cls, model_size: str = "7b") -> ModelConfig:
        """
        Create configuration for Llama models

        Args:
            model_size: Model size (7b, 13b, 70b)

        Returns:
            Llama model configuration
        """
        model_mapping = {
            "7b": "llama3.2:3b",
            "8b": "llama3.2:8b",
            "13b": "llama2:13b",
            "70b": "llama2:70b"
        }

        model_name = model_mapping.get(model_size, "llama3.2:3b")

        return ModelConfig(
            model_name=model_name,
            backend=ModelBackend.OLLAMA,
            max_tokens=2048,
            temperature=0.05,  # Lower temperature for more consistent repairs
            top_p=0.9,
            context_length=8192 if model_size in ["13b", "70b"] else 4096,
            timeout_seconds=60 if model_size in ["13b", "70b"] else 30
        )
