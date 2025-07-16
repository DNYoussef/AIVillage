# SPDX-License-Identifier: Apache-2.0
"""FederatedClient → Hivemind DHT for secure-aggregate FL & inference.
"""

import hivemind  # pip install hivemind
import torch


class FederatedClient:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 initial_peers: list[str],
                 prefix: str = "aivillage"):
        self.model = model
        self.optim = optimizer
        # start DHT
        self.dht = hivemind.DHT(initial_peers=initial_peers, start=True)
        # replicate
        self.parallel = hivemind.ReplicatedDataParallel(
            models=self.model,
            optimizers=self.optim,
            dht=self.dht,
            prefix=prefix
        )

    def train_step(self, batch: dict) -> float:
        """Runs one secure-aggregate batch step."""
        return self.parallel.batch_step(batch)

    def infer(self, batch: dict) -> torch.Tensor:
        """Runs a forward pass under DDP."""
        return self.parallel.forward_step(batch)

    def shutdown(self):
        self.dht.shutdown()

