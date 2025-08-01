import os

from chromadb import PersistentClient
from torch.utils.data import Dataset


class PersonalDataset(Dataset):
    """Return dummy dataset for federated training."""

    def __init__(self, user_id: str, max_tokens: int = 2048):
        home = os.getenv("TWIN_HOME", os.path.expanduser("~/.twin_chroma"))
        self.client = PersistentClient(path=home)
        coll = self.client.get_or_create_collection(f"user:{user_id}")
        res = coll.get(include=["documents"])
        self.data = [d for d in res["documents"] if d][:max_tokens]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def personal_ds(user_id: str, max_tokens: int = 2048) -> PersonalDataset:
    return PersonalDataset(user_id, max_tokens=max_tokens)
