import os
import torch
from . import UltralyticsModel
from .backbone import Backbone
from .encoder import ViTEncoder
from .decoder import ViTDecoder
from .tokenizer import Tokenizer
from .trainer import Trainer, save_pred

class Model(torch.nn.Module):
    def __init__(self,
                 tokenizer:Tokenizer,
                 model:UltralyticsModel|None=None,
                 imgsz:int=640,
                 dim:int=512,
                 encoder_depth:int=3,
                 decoder_depth:int=3,
                 encoder_num_heads:int=8,
                 decoder_num_heads:int=8,
                 dropout:float=0.1,
                 freeze_backbone:bool=True,
                 device:torch.device=torch.device('cpu'),
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocap_size = tokenizer.vocap_size
        self.imgsz = max(64, 32 * (imgsz // 32))
        self.dim = dim
        self.pad_id = tokenizer.char2idx[tokenizer.PAD]
        self.bos_id = tokenizer.char2idx[tokenizer.BOS]
        self.eos_id = tokenizer.char2idx[tokenizer.EOS]
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        self.device = device

        self.backbone = Backbone(model=model, 
                                 imgsz=self.imgsz, 
                                 device=self.device)  # p3/p4/p5 inf
        
        self.encoder  = ViTEncoder(dim=self.dim, 
                                   in_shape=self.backbone.bb_out_shape,
                                   num_heads=encoder_num_heads,
                                   num_blocks=encoder_depth,
                                   dropout=dropout,
                                   device=self.device)

        self.decoder = ViTDecoder(vocab_size=self.vocap_size,
                                  pad_id=self.pad_id,
                                  bos_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  dim=self.dim,
                                  depth=decoder_depth,
                                  heads=decoder_num_heads,
                                  dropout=dropout,
                                  device=self.device)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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
                             freeze_backbone:bool=False,
                             device=torch.device('cpu')):
        from ultralytics import YOLO
        
        ckpt = torch.load(path, map_location=device)
        sd = ckpt.get("model_state_dict", None)
        imgsz = ckpt.get("imgsz",640)
        dim = ckpt.get("decoder_embed_dim", 512)
        encoder_depth = ckpt.get("encoder_depth", 3)
        decoder_depth = ckpt.get("decoder_depth", 3)
        encoder_num_heads = ckpt.get("encoder_num_heads", 8)
        decoder_num_heads = ckpt.get("decoder_num_heads", 8)
        dropout = ckpt.get("dropout", 0.1)
        freeze_backbone = ckpt.get("freeze_backbone", False)
        vocab = ckpt.get("vocab", None)
        model_name = ckpt.get("model_name", "yolo11n.pt")
        yolo_model = YOLO(model_name)

        model = cls(
            tokenizer=tokenizer.set_vocab(vocab), # load training vocabulary
            model=yolo_model,
            imgsz=imgsz,
            dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
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
        dummy = torch.rand(1, 3, self.imgsz, self.imgsz, device=self.device)
        self.train(False)
        torch.onnx.export(self, (dummy), path, dynamo=True)
        model = onnx.load(path)
        model, check = onnxsim.simplify(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, path)
