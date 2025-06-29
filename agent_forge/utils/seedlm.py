import torch
from typing import List, Tuple


def find_best_seed(block: torch.Tensor, candidate_seeds: List[int]) -> Tuple[int, torch.Tensor]:
    best_err, best_seed, best_c = float('inf'), None, None
    flat = block.view(-1, 1)
    for seed in candidate_seeds:
        torch.manual_seed(seed)
        R = torch.randn_like(flat)
        c, _ = torch.lstsq(flat, R).solution[:R.size(1)]
        c_q = torch.round(c * 7) / 7
        recon = (R @ c_q).view(block.shape)
        err = torch.norm(block - recon).item()
        if err < best_err:
            best_err, best_seed, best_c = err, seed, c_q
    return best_seed, best_c


def regenerate_block(seed: int, coeffs: torch.Tensor, shape):
    torch.manual_seed(seed)
    R = torch.randn(torch.prod(torch.tensor(shape)), coeffs.size(0))
    return (R @ coeffs).view(shape)
