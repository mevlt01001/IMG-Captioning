import math
import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d

class CNNEncoder(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 seq_len:int=50, 
                 device=torch.device('cpu')):
        
        super(CNNEncoder, self).__init__()
        self.out_ch = backbone.out_ch
        self.grid_sizes = backbone.grid_sizes
        self.seq_len = seq_len
        self.convs = torch.nn.ModuleList(
            Conv2d(ch, seq_len, 1) 
            for ch in self.out_ch
            ).to(device=device)
        self.pools = torch.nn.ModuleList(
            MaxPool2d(math.ceil(gs/16))
            for gs in self.grid_sizes
            ).to(device=device)
        self.BNs = torch.nn.ModuleList(
            BatchNorm2d(seq_len)
            for i in range(len(self.out_ch))
            ).to(device=device)        
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, backbone.imgsz, backbone.imgsz, device=device)
            out = backbone.forward(dummy)
            out = self.forward(out)
            self.feats = out.shape[-1]
        
        del dummy, out

    def forward(self, x:list[torch.Tensor]):
        # x is backbone output [[B, C, H, W]... p3, p4, p5, ...]
        B = x[0].shape[0]
        out = []
        for idx, feat in enumerate(x):
            # feat.shape is [B, C, H, W]
            _out = self.convs[idx](feat) # CONV: 
            _out = self.BNs[idx](_out)
            _out = self.pools[idx](_out)
            _out = torch.reshape(_out, (B, self.seq_len, -1))
            out.append(_out)
        out = torch.cat(out, dim=-1)
        return out