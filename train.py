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
model = Model.load_from_checkpoint(
    path="Trained_pt/best.pt", 
    tokenizer=tokenizer,
    freeze_backbone=True,
    device=device)

# yolo11n = YOLO("yolo11n.pt")
# model = Model(
#     tokenizer=tokenizer,
#     model=yolo11n,
#     imgsz=224,
#     dim=64,
#     encoder_depth=2,
#     decoder_depth=2,
#     encoder_num_heads=2,
#     decoder_num_heads=2,
#     dropout=0.1,
#     freeze_backbone=False,
#     device=device,
#     use_raw_patches=False,
#     patch_size=None
# )

# model.train(
#     imagepaths=img_paths,
#     epoch=50, 
#     batch_size=16,
#     lr=1e-3,
#     weight_decay=1e-2,
#     grad_clip=1.0,
#     save_dir="train_outs",
#     max_len = None
# )


# To Predict
model.predict("dog_bike_car.jpg")

# To Export onnx format
# model.export(),exit()
