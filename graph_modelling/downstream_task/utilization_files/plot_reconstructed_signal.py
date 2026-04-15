import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import Config
from model import ECGModel
import os
import re
import json
import math
import numpy as np
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader


from BRP_dataset_caregiver import BRPGraphDataset
from classifier import GraphClassifier, load_pretrained_encoder
from supervised_model import SupervisedECGGraph
from train import run_one_epoch
from config import Config

cfg = Config()

pretrained_ckpt = "/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/best_model.pt"
encoder = load_pretrained_encoder(pretrained_ckpt, cfg=cfg, device="cpu")
train_dataset = BRPGraphDataset("../../BRP_train.csv", window_sec=10, sample_rate=1000, cfg=cfg)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

counter = 0
def plot_beats(b1, b2, id):
    x_l = list(range(len(b1)))
    plt.figure(figsize=(20, 10))
    plt.plot(x_l, b1, color = 'b', label="Original")
    plt.plot(x_l, b2, color = 'r', label="reconstructed")
    plt.savefig("id.png", bbox_inches="tight")
    
for i, batch in enumerate(train_loader):
    beats = batch["beats"]
    rr = batch["rr"]
    valid_mask = batch["valid_mask"]
    print(beats.shape)

    x = encoder(
        beats,
        rr,
        node_mask=None,
        valid_mask=valid_mask,
        return_latent=True,
    )
    plot_beats(beats, x, i)
    if counter >3:
        break

