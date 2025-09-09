import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from modules.model import Model
from modules.tokenizer import Tokenizer
from datasets import load_dataset

# CAPTIONS_PATH = "/home/neuron/datasets/obss-intern-competition-2025/train.csv"
# IMAGES_PATH   = "/home/neuron/datasets/obss-intern-competition-2025/train_images/"

ds = load_dataset("ituperceptron/image-captioning-turkish", split="short_captions")

# labels    = pd.read_csv(CAPTIONS_PATH)
captions  = ds["text"]
img_paths = ds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer(captions)

# To Load From Checkpoint
# model = Model.load_from_checkpoint(
#     path="train_outs/last.pt", 
#     tokenizer=tokenizer,
#     freeze_backbone=True,
#     device=device)

model = Model(
    tokenizer=tokenizer,
    model=YOLO("yolo11n.pt"),
    imgsz=480,
    dim=344,
    encoder_depth=3,
    decoder_depth=3,
    encoder_num_heads=8,
    decoder_num_heads=8,
    dropout=0.1,
    freeze_backbone=True,
    device=device
)

model.train(
    imagepaths=img_paths,
    epoch=75, 
    batch_size=64,
    lr=1e-4,
    weight_decay=1e-2,
    grad_clip=1.0,
    save_dir="train_outs",
    max_len = 75
)


# To Predict
# img = np.array(Image.open("images.jpeg").convert("RGB").resize((model.imgsz, model.imgsz)))
# img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
# model.predict(img.to(device))

# To Export onnx format
# model.export(),exit()
