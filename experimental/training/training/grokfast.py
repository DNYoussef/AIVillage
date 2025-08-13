from collections import deque
from typing import Literal

import torch
from langroid import ChatAgent, ChatAgentConfig, Task


class GrokFastTask(Task):
    def __init__(
        self,
        agent: ChatAgent,
        model: torch.nn.Module,
        method: Literal["MA", "EMA"] = "EMA",
        window_size: int = 100,
        lamb: float = 2.0,
        alpha: float = 0.98,
    ) -> None:
        super().__init__(agent)
        self.model = model
        self.method = method
        self.window_size = window_size
        self.lamb = lamb
        self.alpha = alpha
        self.grads = None

    async def filter_gradients(self):
        if self.method == "MA":
            return await self._filter_ma()
        return await self._filter_ema()

    async def _filter_ma(self) -> None:
        if self.grads is None:
            self.grads = {
                n: deque(maxlen=self.window_size)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.grads[n].append(p.grad.data.detach())

                if len(self.grads[n]) == self.window_size:
                    avg = sum(self.grads[n]) / self.window_size
                    p.grad.data = p.grad.data + avg * self.lamb

    async def _filter_ema(self) -> None:
        if self.grads is None:
            self.grads = {
                n: p.grad.data.detach()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.grads[n] = self.grads[n] * self.alpha + p.grad.data.detach() * (
                    1 - self.alpha
                )
                p.grad.data = p.grad.data + self.grads[n] * self.lamb

    async def run(self) -> str:
        await self.filter_gradients()
        return "Gradients filtered successfully"


# Usage example
if __name__ == "__main__":
    import asyncio

    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main() -> None:
        config = ChatAgentConfig(
            name="GrokFastAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        model = torch.nn.Linear(10, 10)  # Example model
        task = GrokFastTask(agent, model)
        result = await task.run()
        print(result)

    asyncio.run(main())
