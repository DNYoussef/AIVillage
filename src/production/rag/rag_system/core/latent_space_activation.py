class LatentSpaceActivation:
    def __init__(self) -> None:
        self.latent_space = {}

    async def activate(self, content: str, embeddings: list, entities: list, relations: list) -> str:
        self.latent_space[content] = {
            "embedding": embeddings,
            "entities": entities,
            "relations": relations,
        }
        return f"Activated knowledge for: {content}"

    async def evolve(self) -> str:
        for data in self.latent_space.values():
            if isinstance(data.get("embedding"), list) and data["embedding"]:
                mean_val = sum(data["embedding"]) / len(data["embedding"])
                data["embedding"] = [mean_val for _ in data["embedding"]]
        return "Latent space evolved"
