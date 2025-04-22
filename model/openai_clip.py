import torch
import clip
from PIL import Image

def openai_clip(backbone):
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    return model, preprocess
    