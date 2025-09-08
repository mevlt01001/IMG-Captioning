import os, math, json, time, random
from dataclasses import dataclass
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from .tokenizer import Tokenizer

def _load_img(path, imgsz):
    if not isinstance(path, str): # PIL
        img = path.convert("RGB").resize((imgsz, imgsz))
        data = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float() / 255.0            
        return data

    assert os.path.exists(path)
    img = np.array(Image.open(path).convert("RGB").resize((imgsz, imgsz)))    
    data = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    return data # [B, C, H, W]

def load_images(paths, imgsz, device=torch.device('cpu')):
    if isinstance(paths, dict): # For huggingface datasets my example "ituperceptron/image-captioning-turkish"
        try:
            return torch.cat([_load_img(p, imgsz) for p in paths["image"]], dim=0).to(device=device)
        except Exception as e:
            print(e)
            print("If you are using Huggingface dataset or something else, please implement your own Trainer.load_images and Trainer._load_img function.")
            exit(1)
    elif isinstance(paths[0], dict):
        return torch.cat([_load_img(p["image"], imgsz) for p in paths], dim=0).to(device=device)

    return torch.cat([_load_img(p, imgsz) for p in paths], dim=0).to(device=device)

def load_captions(captions:list[list[int]], vocap_size:int, device=torch.device('cpu')):
    tin = []
    tout = []
    for cap in captions:
        tin.append(cap[:-1])
        tout.append(cap[1:])
    
    tin = torch.tensor(tin).long().to(device=device)
    tout = torch.tensor(tout).long().to(device=device)
    return tin, tout

def remove_accents(text: str) -> str:    
    normalized = unicodedata.normalize("NFKD", text)    
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def turkish_ascii(text: str) -> str:
    text = remove_accents(text)
    return text.replace("ı","i").replace("İ","I")

def save_pred(img: torch.Tensor, caption: str, save_path: str = "pred.png"):
    if img.ndim != 3 or img.shape[0] not in (1,3):
        raise ValueError(f"img shape must be [C,H,W] with C=1 or 3, got {tuple(img.shape)}")

    C, H, W = img.shape
    imgsz = W
    caption = turkish_ascii(caption)
    if C == 1:
        img = img.repeat(3, 1, 1)
    img_np = (img.clamp(0,1).permute(1,2,0) * 255.0).cpu().numpy().astype(np.uint8)

    panel_pil = Image.new("RGB", (imgsz, imgsz), (0,0,0))
    draw = ImageDraw.Draw(panel_pil)
    font = ImageFont.load_default(size=16)
    wrapped = textwrap.fill(caption, width=16)

    x0, y0 = 24, 24
    bbox = draw.multiline_textbbox((x0, y0), wrapped, font=font, spacing=6)
    pad = 12
    
    draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=(0,0,0))
    draw.multiline_text((x0, y0), wrapped, font=font, fill=(255,255,255),
                        spacing=6, stroke_width=2, stroke_fill=(0,0,0))
    
    panel_np = np.array(panel_pil)
    frame = np.concatenate([img_np, panel_np], axis=1)

    Image.fromarray(frame).save(save_path)
    

@dataclass
class TrainConfig:
    epoch:int = 20
    batch_size:int = 64
    lr:float = 2e-4
    weight_decay:float = 1e-2
    grad_clip:float = 1.0
    num_workers:int = 4
    save_dir:str = "checkpoints"


class Trainer:
    def __init__(self, model:nn.Module, device:torch.device=torch.device('cpu')):
        self.model  = model
        self.device = device    

    @torch.no_grad()
    def validate(self, 
                 imagepaths:list[str], 
                 tokenizer:Tokenizer
                 ) -> str:
        images = load_images(imagepaths, self.model.imgsz, device=self.device) # [B, C, H, W]
        preds = []
        for img in images:
            p = self.model.forward(img.unsqueeze(0)) # [B, S]
            preds.append(p.squeeze(0))
        
        for i,c in enumerate(preds):
            cap = tokenizer.decode(c.tolist())
            os.makedirs(f"{self.cfg.save_dir}/preds", exist_ok=True)
            save_pred(images[i], cap, os.path.join(f"{self.cfg.save_dir}/preds", f"{i}.png"))

    def fit(self,
            tokenizer:Tokenizer,
            imagepaths:list[str],
            epoch:int=20,
            batch_size:int=64,
            lr:float=2e-4,
            weight_decay:float=1e-2,
            grad_clip:float=1.0,
            save_dir:str="checkpoints",
            ):
        
        self.cfg = TrainConfig(epoch=epoch,
                          batch_size=batch_size,
                          lr=lr,
                          weight_decay=weight_decay,
                          grad_clip=grad_clip,
                          save_dir=save_dir)
        cfg = self.cfg
        
        os.makedirs(cfg.save_dir, exist_ok=True)

        self.captions = tokenizer.tokenized_captions

        global_step = 0
        total_step = cfg.epoch*math.ceil(len(self.captions)/cfg.batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_step)
        criterion = nn.CrossEntropyLoss(ignore_index=int(self.model.pad_id))

        best_loss = float("inf")
        self.model.to(self.device)
        self.model.train(True)
        scaler = GradScaler(enabled=True)

        for ep in range(1, cfg.epoch+1):
            global_loss = 0.0
            seen = 0
            t0 = time.time()

            for batch_start_idx in range(0, len(self.captions), cfg.batch_size):
                batch_size = min(cfg.batch_size, len(self.captions)-batch_start_idx)

                paths = imagepaths[batch_start_idx:batch_start_idx+batch_size]
                captions = self.captions[batch_start_idx:batch_start_idx+batch_size]

                imgs = load_images(paths, imgsz=self.model.imgsz, device=self.device)
                tins, touts = load_captions(captions, self.model.vocap_size, device=self.device) # 2x [B, T, V]
                pad_id = self.model.pad_id
                valid_lens = (touts != pad_id).sum(dim=1)              # [B]
                batch_max_len = int(valid_lens.max().item())
                tins  = tins[:,  :batch_max_len]
                touts = touts[:, :batch_max_len]
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model.forward(imgs, tins)
                    loss = criterion.forward(logits.reshape(-1, logits.size(-1)), touts.reshape(-1))

                opt.zero_grad()
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()

                bsz = imgs.size(0)
                global_loss += loss.item() * bsz
                seen += bsz
                global_step += 1
                print(f"[%{100*((batch_start_idx+batch_size)/len(self.captions)):>6.2f}]Epoch {ep:03d} loss={global_loss/seen:<6.3f} lr={sched.get_last_lr()[0]:<.6f}", end="\r")

            random_idx:list = np.random.randint(0, len(self.captions), cfg.batch_size).tolist()
            paths = []
            for idx in random_idx:
                paths.append(imagepaths[idx])
            
            self.model.train(False)
            self.validate(paths, tokenizer)
            self.model.train(True)

            ep_loss = global_loss/seen
            dt = time.time()-t0
            print(f"[EPOCH {ep:03d}] loss={ep_loss:.4f}  time={dt:.1f}s")

            ckpt = {
                "model_name": self.model.backbone.model_name,
                "model_state_dict": self.model.state_dict(),
                "vocab": tokenizer.vocap,
                "last_avg_loss": ep_loss,
                "pad_id": self.model.pad_id, 
                "bos_id": self.model.bos_id, 
                "eos_id": self.model.eos_id,
                "imgsz": self.model.imgsz,
                "LSTM_num_layers": self.model.num_layers,
                "LSTM_hidden_feats": self.model.hidden_feats,
                "decoder_embed_dim": self.model.dim
            }

            last_path = os.path.join(cfg.save_dir, "last.pt")
            torch.save(ckpt, last_path)
            if ep_loss < best_loss:
                best_loss = ep_loss
                torch.save(ckpt, os.path.join(cfg.save_dir, "best.pt"))

        print(f"[TRAIN DONE] best_loss={best_loss:.4f}  ckpts => {cfg.save_dir}")
        return {"best_loss": best_loss, "save_dir": cfg.save_dir}
