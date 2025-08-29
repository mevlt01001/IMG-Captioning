import os
import pandas as pd
import torch
from ultralytics import YOLO
from modules.model import Model
from modules.tokenizer import Tokenizer

CAPTIONS_PATH = "/home/neuron/datasets/obss-intern-competition-2025/train.csv"
IMAGES_PATH   = "/home/neuron/datasets/obss-intern-competition-2025/train_images/"

labels    = pd.read_csv(CAPTIONS_PATH)
captions  = labels["caption"].astype(str).tolist()
img_paths = labels["image_id"].apply(lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg")).tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer(captions)

model = Model(
    tokenizer=tokenizer,
    model=YOLO("yolo11n.pt"),
    imgsz=640,
    dim=512,
    hidden_feats=512,
    num_layers=5,
    device=device
)

model.train(
    imagepaths=img_paths,
    epoch=100, 
    batch_size=16, 
    lr=2e-4, 
    weight_decay=1e-2, 
    grad_clip=1.0,
    save_dir="train_outs",
)

# To Load From Checkpoint
# model = Model.load_from_checkpoint(path="train_outs/best.pt", tokenizer=tokenizer, device=device)

# To Export onnx format
# model.export(),exit()