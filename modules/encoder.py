import torch
from torch.nn import Conv2d

class Encoder(torch.nn.Module):
    def __init__(self, backbone, out_ch:int=512):
        super(Encoder, self).__init__()
        self.out_ch = out_ch
        self.convs = torch.nn.ModuleList(
            [Conv2d(in_ch, out_ch, 1) for in_ch in backbone.out_ch]).to(backbone.device)

    def forward(self, x:list[torch.Tensor]):
        # x is backbone output
        B = x[0].shape[0]
        out = []
        for i, xx in enumerate(x):
            out.append(self.convs[i](xx).reshape(B, self.out_ch, -1).permute(0, 2, 1))
        
        return torch.cat(out, dim=1)
