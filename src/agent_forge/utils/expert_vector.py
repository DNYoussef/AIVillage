import torch


class ExpertVector:
    def __init__(self, name: str, svs: dict[str, torch.Tensor]) -> None:
        self.name = name
        self.svs = svs

    def apply(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        for layer_name, z in self.svs.items():
            layer = modules[layer_name]
            W = layer.weight.data
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            S_new = S * z.to(S.device)
            layer.weight.data = U @ torch.diag(S_new) @ Vh
