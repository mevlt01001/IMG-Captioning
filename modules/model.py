import torch
from .backbone import Backbone
from .encoder import Encoder
from . import UltralyticsModel
from ultralytics import YOLO

class Model(torch.nn.Module):
    def __init__(self, 
                 model:UltralyticsModel=YOLO("yolo11m.pt"),
                 imgsz:int= 640,
                 dim:int=512
                 ):
        
        super(Model, self).__init__()
        self.backbone = Backbone(model, imgsz)
        self.encoder = Encoder(self.backbone, dim)
        print(f"Backbone:\n\t{self.backbone}")
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.encoder(x)
        return x


