import math
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):

    def __init__(self, 
                 backbone, 
                 proj_dim:int,
                 device=torch.device('cpu')
                 ):
        
        super().__init__()
        self.backbone = backbone
        self.D = proj_dim

        self.proj = nn.ModuleList([nn.Conv2d(c, proj_dim, kernel_size=1) for c in backbone.out_ch]).to(device=device)
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(math.ceil(g/2)) for g in backbone.grid_sizes]).to(device=device)
        self.ln   = nn.LayerNorm(proj_dim).to(device=device)

        self.S = sum(math.ceil(g/2)**2 for g in backbone.grid_sizes)
        self.feats = proj_dim  # D

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [p3, p4, p5] each one is [B, C, H, W]
        seqs = []
        for x, conv, pool in zip(feats, self.proj, self.pool):
            z = conv(x)                    # [B, D, H, W]
            z = pool(z)                    # [B, D, h, w]
            z = z.flatten(2).transpose(1, 2)  # [B, D, S] -> [B, S, D]
            z = self.ln(z)
            seqs.append(z)
        img_seq = torch.cat(seqs, dim=1)   # [B, S, D]
        return img_seq
