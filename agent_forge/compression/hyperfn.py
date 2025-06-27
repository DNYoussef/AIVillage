import torch
import math
from typing import Dict, List

class HyperCompressionEncoder:
    def __init__(self, num_clusters: int = 16):
        self.num_clusters = num_clusters

    def _cluster(self, weights: torch.Tensor) -> List[Dict]:
        flat = weights.flatten()
        idx = torch.argsort(flat.abs())
        clusters=[]
        size = len(flat)//self.num_clusters
        for i in range(self.num_clusters):
            s=i*size
            e=s+size if i<self.num_clusters-1 else len(flat)
            indices=idx[s:e]
            cluster=flat[indices]
            clusters.append({'weights':cluster,'indices':indices})
        return clusters

    def _search_params(self, w: torch.Tensor) -> Dict:
        mean=w.mean().item()
        best={'A':0,'B':0,'C':0,'D':mean,'an':1,'ad':2,'err':float('inf')}
        A_range=torch.linspace(-2,2,4)
        B_range=torch.linspace(-2,2,4)
        for A in A_range:
            for B in B_range:
                for alpha_n in [1,2,3]:
                    for alpha_d in [2,3,5]:
                        theta=2*math.pi*(alpha_n/alpha_d)*torch.arange(len(w))
                        traj=A*torch.sin(theta)+B*torch.cos(theta)+mean
                        err=torch.sum((w-traj)**2).item()
                        if err<best['err']:
                            best.update({'A':A.item(),'B':B.item(),'C':0,'D':mean,'an':alpha_n,'ad':alpha_d,'err':err})
        return best

    def compress_weight_matrix(self, weight_matrix: torch.Tensor) -> Dict:
        clusters=self._cluster(weight_matrix)
        params=[self._search_params(c['weights']) for c in clusters]
        return {'params':params,'original_shape':weight_matrix.shape}

    def decompress_weight_matrix(self, data: Dict) -> torch.Tensor:
        shape=data['original_shape']
        total=shape.numel() if isinstance(shape,torch.Size) else int(torch.prod(torch.tensor(shape)))
        out=torch.zeros(total)
        for p in data['params']:
            idx=p['indices']
            n=len(idx)
            alpha=p['an']/p['ad']
            theta=2*math.pi*alpha*torch.arange(n)
            traj=p['A']*torch.sin(theta)+p['B']*torch.cos(theta)+p['D']
            out[idx]=traj
        return out.reshape(shape)
