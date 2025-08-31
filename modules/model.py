import os
import torch
from . import UltralyticsModel
from .backbone import Backbone
from .encoder import CNNEncoder
from .decoder import LSTMDecoder
from .tokenizer import Tokenizer
from .trainer import Trainer, save_pred

class Model(torch.nn.Module):
    def __init__(self,
                 tokenizer:Tokenizer,
                 model:UltralyticsModel|None=None,
                 imgsz:int=640,
                 dim:int=512,
                 hidden_feats:int=512,
                 num_layers:int=3,
                 device:torch.device=torch.device('cpu'),
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device

        self.imgsz = max(64, 32 * (imgsz // 32))
        self.dim = dim
        self.hidden_feats = hidden_feats
        self.num_layers = num_layers

        self.vocap_size = tokenizer.vocap_size
        self.pad_id = tokenizer.char2idx[tokenizer.PAD]
        self.bos_id = tokenizer.char2idx[tokenizer.BOS]
        self.eos_id = tokenizer.char2idx[tokenizer.EOS]

        self.backbone = Backbone(model=model, imgsz=self.imgsz, device=self.device)  # p3/p4/p5 çıkarımı
        self.encoder  = CNNEncoder(backbone=self.backbone, proj_dim=self.dim, device=self.device)
        self.feats    = self.encoder.feats

        self.decoder = LSTMDecoder(
            vocab_size=self.vocap_size,
            token_feats=self.feats,
            hidden_size=self.hidden_feats,
            num_layer=self.num_layers,
            D_img=self.dim,
            pad_id=self.pad_id,
            bos_id=self.bos_id, 
            eos_id=self.eos_id,
            device=self.device
        )

    @torch.no_grad()
    def predict(self, images:torch.Tensor):
        self.eval()
        assert images.ndim == 4, "x must be [B,C,H,W]"
        pred = self.forward(images)
        B = pred.size(0)
        for i in range(B):
            cap = self.tokenizer.decode(pred[i].tolist())
            os.makedirs(f"Inference_outs/preds", exist_ok=True)
            save_pred(images[i], cap, os.path.join(f"Inference_outs/preds", f"{i}.png"))

    def forward(self, x, tokens_in=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder.forward(x, tokens_in) if self.training else self.decoder.generate(x, max_len=self.tokenizer.seq_len)
        return x
    
    @classmethod
    def load_from_checkpoint(cls, 
                             path:str, 
                             tokenizer:Tokenizer, 
                             device=torch.device('cpu')):
        from ultralytics import YOLO
        
        ckpt = torch.load(path, map_location=device)
        sd = ckpt.get("model_state_dict", None)
        imgsz = ckpt.get("imgsz",640)
        dim = ckpt.get("decoder_embed_dim", 512)
        hidden_feats = ckpt.get("LSTM_hidden_feats", 512)
        num_layers = ckpt.get("LSTM_num_layers", 3)
        vocab = ckpt.get("vocab", None)
        model_name = ckpt.get("model_name", "yolo11n.pt")
        yolo_model = YOLO(model_name)

        model = cls(
            tokenizer=tokenizer.set_vocab(vocab), # load training vocabulary
            model=yolo_model,
            imgsz=imgsz,
            dim=dim,
            hidden_feats=hidden_feats,
            num_layers=num_layers,
            device=device
        )

        res = model.load_state_dict(sd, strict=False)
        print(res.missing_keys)
        print(res.unexpected_keys)

        return model
        
    def train(self, mode:bool=True, **kwargs):
        has_fit_args = ("imagepaths" in kwargs)
        if not has_fit_args:
            return super().train(mode)

        super().train(True)
        trainer = Trainer(model=self, device=self.device)
        return trainer.fit(**kwargs, tokenizer=self.tokenizer)

    def export(self, path:str="model.onnx"):
        import onnx, onnxsim
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
        torch.onnx.export(self, dummy, path, opset_version=19)
        model = onnx.load(path)
        model, check = onnxsim.simplify(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, path)
