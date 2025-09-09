import math
import torch
import torch.nn as nn

@torch.no_grad()
def sinusoidal_2d_pe(H: int, W: int, D: int, device=None) -> torch.Tensor:

    assert D % 4 == 0, f"D % 4 == 0 must; D={D}"
    device = device or torch.device('cpu')

    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # [H,W]

    d = D // 4
    k = torch.arange(d, device=device, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * k / d)  # [d]

    # Broadcast: [H,W,1]*[d] -> [H,W,d]
    y_sin = torch.sin(yy[..., None] * omega)  # [H,W,d]
    y_cos = torch.cos(yy[..., None] * omega)  # [H,W,d]
    x_sin = torch.sin(xx[..., None] * omega)  # [H,W,d]
    x_cos = torch.cos(xx[..., None] * omega)  # [H,W,d]

    pe = torch.cat([y_sin, y_cos, x_sin, x_cos], dim=-1)  # [H,W,D]
    pe = pe.view(1, H*W, D).contiguous()
    return pe

class EncoderAttnBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads:int=8,
                 dropout:float=0.2,
                 mlp_ratio:float=4.0):
        
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: [B,S,D]
        h = self.norm1(x) 
        x = x + self.attn.forward(h,h,h, need_weights=False)[0] # [B,S,D]
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class ViTEncoder(torch.nn.Module):
    def __init__(self,
                 dim:int,
                 in_shape:list[int],
                 num_blocks=2, 
                 num_heads=8, 
                 dropout=0.1,
                 device=torch.device('cpu')):
        
        super().__init__()
        self.H = in_shape[2]
        self.W = in_shape[3]
        self.blocks = torch.nn.ModuleList([
            EncoderAttnBlock(dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_blocks)
        ]).to(device=device)
        self.norm = torch.nn.LayerNorm(dim).to(device=device)
        self.proj = nn.Conv2d(in_shape[1], dim, 1).to(device=device)
        self.ln = nn.LayerNorm(dim).to(device=device)

    def forward(self, feats:torch.Tensor):      #feats: [B,C,H,W]
        feats = self.proj.forward(feats)        # [B, dim, H, W]
        feats = feats.flatten(2)                # [B, dim, S]
        feats = feats.transpose(1, 2)           # [B, S, dim]
        vis_tokens = self.ln.forward(feats)     # [B, S, dim]

        B,S,D = vis_tokens.shape
        pe = sinusoidal_2d_pe(H=self.H, 
                              W=self.W, 
                              D=D, 
                              device=vis_tokens.device)
        x = vis_tokens + pe # Positional Encoding
        
        # Attention Blocks
        for block in self.blocks:
            x = block(x) 

        return self.norm(x)         # [B,S,D]

class CNNEncoder(nn.Module):

    def __init__(self,
                 dim:int,
                 in_shape:list[int],
                 device=torch.device('cpu')
                 ):
        
        super().__init__()

        self.conv = nn.Conv2d(in_shape[1], dim, 1).to(device=device)
        self.ln = nn.LayerNorm(dim).to(device=device)
        self.visual_patch = in_shape[-1] * in_shape[-2]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
                                            # feats: [B, C, H, W]
        feats = self.conv.forward(feats)    # [B, dim, H, W]
        feats = feats.flatten(2)            # [B, dim, visual_patch]
        feats = feats.transpose(1, 2)       # [B, visual_patch, dim]
        feats = self.ln.forward(feats)      # [B, visual_patch, dim]
        return feats
