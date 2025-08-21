import torch
from ingestion.vector_ds import personal_ds
from src.communications.federated_client import FederatedClient
from twin_runtime.fine_tune import run_nightly
from twin_runtime.runner import LLM


def nightly(user_id: str) -> None:
    ds = personal_ds(user_id, max_tokens=2048)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    opt = torch.optim.AdamW(LLM._model.parameters(), lr=1e-5)
    fed = FederatedClient(
        LLM._model,
        opt,
        initial_peers=["/dns4/bootstrap.mesh/tcp/4001/p2p/QmX"],
    )
    for _ in range(1):
        for batch in loader:
            fed.train_step(batch)
    run_nightly(user_id)
    fed.shutdown()
