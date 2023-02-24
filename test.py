import torch
from modules import UNet
from ddpm import Diffusion
import os
from PIL import Image
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from Other_functions import *
from utils import *



import argparse    
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
args.image_size = 64
args.batch_size = 10
args.dataset_path = "animal10/raw-img"

dataloader = get_data_weight_random_sampler(args)

# get a batch of images from the data loader

images, labels = next(iter(dataloader))
images = (images+1)/2

print(labels)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.axis('off')
plt.show()




