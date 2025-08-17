from . import heads
from . import UltralyticsModel
from . import Concat
import numpy as np
import torch

class Backbone(torch.nn.Module):
    """
    This class extracts 'ultralytics.engine.model.Model's features pyramid (FPN) (p3, p4, p5,...) output from given model as a backbone.
    """
    def __init__(self, model: UltralyticsModel, imgsz:int=640):
        super(Backbone, self).__init__()
        self.model_name = model.model_name
        self.device = model.device
        self.detect_feats_from = model.model.model[-1].f # list of feature map layer indices [..., p3, p4, p5, ...]
        self.layers = torch.nn.ModuleList(model.model.model[:-1]).to(self.device)
        self.imgsz = imgsz
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device)
            out = self.forward(dummy)

        self.out_ch = [p.shape[1] for p in out]

    def __str__(self):
        return f"""
        model: {self.model_name} at {self.device}
        imgsz: {self.imgsz}
        feat_from: {self.detect_feats_from}
        feat_ch: {self.out_ch}
        """
    
    def forward(self, x):
        outputs = []
        for m in self.layers:
            if isinstance(m, heads ):
                return [outputs[f] for f in self.detect_feats_from]
            elif isinstance(m, Concat):
                x = m([outputs[f] for f in m.f])
            else:
                x = m(x) if m.f == -1 else m(outputs[m.f])
            outputs.append(x)
        return [outputs[f] for f in self.detect_feats_from]
    
