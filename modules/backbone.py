from . import heads
from . import UltralyticsModel
from . import Concat
import torch

class Backbone(torch.nn.Module):
    """
    This class extracts 'ultralytics.engine.model.Model's backbone.
    """
    def __init__(self, 
                 model: UltralyticsModel, 
                 imgsz:int=640,
                 device=torch.device('cpu')):
        
        super(Backbone, self).__init__()
        self.model_name = model.model_name
        self.device = device
        self.layers = torch.nn.ModuleList(model.model.model[:-1]).to(self.device)
        self.imgsz = imgsz
        
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device, requires_grad=False)
        out = self.shape_infer(dummy)
        self.bb_out_shape = out.shape

        del out, dummy
    
    def forward(self, x):
        outputs = []
        for m in self.layers:
            if isinstance(m, heads):
                return outputs[-1]
            elif isinstance(m, Concat):
                x = m([outputs[f] for f in m.f])
            else:
                x = m(x) if m.f == -1 else m(outputs[m.f])
            outputs.append(x)

        feats = outputs[-1]                 # feats: [B, C, H, W]
        return feats
    
    @torch.no_grad()
    def shape_infer(self, x):
        outputs = []
        
        for m in self.layers:
            if isinstance(m, heads):
                return outputs[-1]
            elif isinstance(m, Concat):
                x = m([outputs[f] for f in m.f])
            else:
                x = m(x) if m.f == -1 else m(outputs[m.f])
            outputs.append(x)

        return outputs[-1]
    
