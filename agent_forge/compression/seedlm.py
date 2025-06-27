import torch
import math
from typing import Dict, List, Tuple

class LFSRGenerator:
    """Hardware-friendly LFSR pseudo-random generator"""
    def __init__(self, seed: int, taps: List[int] = None):
        self.register = seed & 0xFFFF
        self.taps = taps or [16, 14, 13, 11]
        self.initial_seed = seed

    def next_bit(self) -> int:
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.register >> (tap - 1)) & 1
        self.register = (self.register >> 1) | (feedback << 15)
        return self.register & 1

    def generate_matrix(self, rows: int, cols: int) -> torch.Tensor:
        matrix = torch.zeros(rows, cols, dtype=torch.float32)
        for i in range(rows):
            for j in range(cols):
                bit = self.next_bit()
                matrix[i, j] = 1.0 if bit else -1.0
        return matrix / math.sqrt(cols)

class SeedLMCompressor:
    def __init__(self, block_size: int = 8, latent_dim: int = 4, num_seeds: int = 256):
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.num_seeds = num_seeds

    def compress_weight_matrix(self, weight_matrix: torch.Tensor) -> Dict:
        flat = weight_matrix.flatten()
        blocks = self._create_blocks(flat)
        compressed = []
        for block in blocks:
            compressed.append(self._compress_single_block(block))
        ratio = self._compression_ratio(weight_matrix, compressed)
        return {
            'compressed_blocks': compressed,
            'original_shape': weight_matrix.shape,
            'compression_ratio': ratio,
        }

    def _create_blocks(self, flat: torch.Tensor) -> List[torch.Tensor]:
        blocks = []
        for i in range(0, len(flat), self.block_size):
            block = flat[i:i+self.block_size]
            if len(block) < self.block_size:
                pad = torch.zeros(self.block_size)
                pad[:len(block)] = block
                block = pad
            blocks.append(block)
        return blocks

    def _compress_single_block(self, block: torch.Tensor) -> Dict:
        best = {'seed':0,'coeff':torch.zeros(self.latent_dim,dtype=torch.int8),'exp':0,'error':float('inf')}
        candidate_seeds = torch.randint(1,2**16,(self.num_seeds,)).tolist()
        for seed in candidate_seeds:
            lfsr = LFSRGenerator(seed)
            basis = lfsr.generate_matrix(self.block_size, self.latent_dim)
            coeff = torch.linalg.lstsq(basis, block).solution
            q, exp = self._quantize(coeff)
            recon = basis @ self._dequantize(q, exp)
            err = torch.sum((block - recon)**2).item()
            if err < best['error']:
                best = {'seed':seed,'coeff':q,'exp':exp,'error':err}
        return best

    def _quantize(self, coeff: torch.Tensor) -> Tuple[torch.Tensor,int]:
        if coeff.numel()==0:
            return torch.zeros(0,dtype=torch.int8),0
        max_abs = coeff.abs().max()
        if max_abs==0:
            return torch.zeros_like(coeff,dtype=torch.int8),0
        exp = max(0,int(torch.log2(max_abs).ceil().item())-3)
        scale = 2**(-exp)
        q = torch.clamp(torch.round(coeff*scale),-8,7).to(torch.int8)
        return q, exp

    def _dequantize(self, q: torch.Tensor, exp: int) -> torch.Tensor:
        return q.float() * (2**exp)

    def _compression_ratio(self, original: torch.Tensor, blocks: List[Dict]) -> float:
        original_bits = original.numel()*32
        bits=0
        for b in blocks:
            bits+=16+4+len(b['coeff'])*4
        return original_bits / bits if bits>0 else 0

    def decompress_weight_matrix(self, data: Dict) -> torch.Tensor:
        blocks=[]
        for b in data['compressed_blocks']:
            lfsr=LFSRGenerator(b['seed'])
            basis=lfsr.generate_matrix(self.block_size,self.latent_dim)
            coeff=self._dequantize(b['coeff'], b['exp'])
            blocks.append(basis @ coeff)
        flat=torch.cat(blocks)[:int(torch.prod(torch.tensor(data['original_shape'])))]
        return flat.reshape(data['original_shape'])
