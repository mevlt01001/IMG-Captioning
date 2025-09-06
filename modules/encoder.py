import math
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):

    def __init__(self,
                 out_ch:list[int],
                 grid_sizes:list[int],
                 proj_dim:int,
                 device=torch.device('cpu')
                 ):
        
        super().__init__()
        self.out_ch = out_ch
        self.grid_sizes = grid_sizes
        self.proj = nn.ModuleList([nn.Conv2d(c, proj_dim, kernel_size=1) for c in self.out_ch]).to(device=device)
        target_sizes = [(g // 2) if (g % 2 == 0) else g for g in self.grid_sizes]
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(t) for t in target_sizes]).to(device=device)
        self.ln   = nn.LayerNorm(proj_dim).to(device=device)

        self.S = sum(t*t for t in target_sizes)
        self.feats = proj_dim  # D

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [p3, p4, p5] each one is [B, C, H, W]
        seqs = []
        for x, conv, pool in zip(feats, self.proj, self.pool):
            z = conv(x)                    # [B, D, H, W]
            z = pool.forward(z)                    # [B, D, h, w]
            z = z.flatten(2).transpose(1, 2)  # [B, D, S] -> [B, S, D]
            z = self.ln(z)
            seqs.append(z)
        img_seq = torch.cat(seqs, dim=1)   # [B, S, D]
        return img_seq
