import torch
from . import UltralyticsModel
from .backbone import Backbone
from .encoder import CNNEncoder
from .decoder import LSTMDecoder
from ultralytics import YOLO

class Model(torch.nn.Module):
    def __init__(self, 
                 model:UltralyticsModel,
                 imgsz:int= 640,
                 seq_len:int=50,
                 vocap_size:int=26000,
                 hidden_feats:int=512,
                 num_layers:int=3,
                 device:torch.device=torch.device('cpu')
                 ):
        
        super(Model, self).__init__()

        self.model = model
        self.imgsz = imgsz
        self.seq_len = seq_len
        self.vocap_size = vocap_size
        self.hidden_feats = hidden_feats
        self.num_layers = num_layers
        self.device = device

        self.backbone = Backbone(
            model=self.model,
            imgsz=self.imgsz,
            device=self.device
        )

        self.encoder = CNNEncoder(
            backbone=self.backbone,
            seq_len=self.seq_len,
            device=self.device
        )

        self.feats = self.encoder.feats

        self.decoder = LSTMDecoder(
            vocap_size=self.vocap_size,
            seq_len=self.seq_len,
            input_feats=self.feats,
            hidden_feats=self.hidden_feats,
            num_layers=self.num_layers,
            device=self.device
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x




