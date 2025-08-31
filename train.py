import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
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

# To Load From Checkpoint
# model = Model.load_from_checkpoint(path="train_outs/best.pt", tokenizer=tokenizer, device=device)

model = Model(
    tokenizer=tokenizer,
    model=YOLO("yolo11n.pt"),
    imgsz=512,
    dim=256,
    hidden_feats=512,
    num_layers=5,
    device=device
)

model.train(
    imagepaths=img_paths,
    epoch=125, 
    batch_size=32, 
    lr=4e-4, 
    weight_decay=1e-2, 
    grad_clip=1.0,
    save_dir="train_outs",
)


# To Predict
# img = np.array(Image.open("test.jpg").convert("RGB").resize((model.imgsz, model.imgsz)))
# img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
# model.predict(img.to(device))

# To Export onnx format
# model.export(),exit()