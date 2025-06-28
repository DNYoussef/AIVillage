import torch, torch.nn as nn, random


class Geo2Z(nn.Module):
    def __init__(self, in_dim=5, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.Tanh(),
            nn.Linear(128,z_dim),  nn.Tanh()
        )

    def forward(self, geom_vec):
        return .05 * self.net(geom_vec)   # bounded ΔΣ


class Replay:
    def __init__(self, N): self.buf,self.N=[],N
    def add(self,*t):
        if len(self.buf)>=self.N: self.buf.pop(0)
        self.buf.append(t)
    def sample(self,k): return random.sample(self.buf, k)
