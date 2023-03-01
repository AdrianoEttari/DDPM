import torch
from modules import UNet_conditional
from ddpm import Diffusion
import os
from PIL import Image
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
# from Other_functions import *
from utils import *



# import argparse    
# parser = argparse.ArgumentParser()
# args, unknown = parser.parse_known_args()
# args.image_size = 64
# args.batch_size = 10
# args.dataset_path = "animal10/raw-img"

# dataloader = get_data_weight_random_sampler(args)

# # get a batch of images from the data loader

# images, labels = next(iter(dataloader))
# images = (images+1)/2

# print(labels)

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(images[i].permute(1, 2, 0))
#     plt.axis('off')
# plt.show()


import imageio
import torchvision
import torch
from PIL import Image

def gif_forward_creator(image_path, time_steps, save_path, alpha_hat, fps=24):
    img_from_data = Image.open(image_path)
    image = torchvision.transforms.ToTensor()(img_from_data)[None, :]
    t = torch.arange(0, time_steps).long()
    noised_image, _ = noise_images(image, t, alpha_hat)
    images = torch.cat([image, noised_image], dim=0)
    pil_images = [torchvision.transforms.ToPILImage()(images[i]) for i in range(time_steps+1)]
    imageio.mimsave(save_path, pil_images, fps=fps)

def prepare_noise_schedule(noise_schedule, beta_start=1e-4, beta_end=0.02, noise_steps=1000):
        if noise_schedule == 'linear':
            beta = torch.linspace(beta_start, beta_end, noise_steps)
            alpha = 1. - beta
            alpha_hat = torch.cumprod(alpha, dim=0) 
            return alpha_hat
        elif noise_schedule == 'cosine':
            f_t = torch.cos(((((torch.arange(noise_steps)/noise_steps)+0.008)/(1+0.008))*torch.pi/2))**2
            alpha_hat = f_t/f_t[0]  
            return alpha_hat

def noise_images(x, t, alpha_hat):
        '''
        This function returns the x_t noise image (check the training algorithm)

        ATTENTION: The error Ɛ is random, but how much of it we add to move forward, depends on the Beta schedule.
        '''
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None] # Each None is a new dimension (e.g.
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)). It doens't add
        # them exatly in the same place, but it adds them in the place where the None is.
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x) # torch.randn_like() returns a tensor of the same shape of x with random values from a standard gaussian (notice that the values inside x are not relevant)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

gif_forward_creator('raw-img/gatto/1.jpeg', 120, 'animations/image_destruction_cosine_120.gif', prepare_noise_schedule('cosine', noise_steps=1000), fps=24)
# gif_forward_creator('raw-img/gatto/1.jpeg', 120, 'animations/image_destruction_linear_120.gif', prepare_noise_schedule('linear', noise_steps=1000), fps=24)

x = torch.arange(1000)
y_linear = prepare_noise_schedule('linear', noise_steps=1000)
y_cosine = prepare_noise_schedule('cosine', noise_steps=1000)

plt.plot(x, y_linear, 'r-', label='linear')
plt.plot(x, y_cosine, 'b-', label='cosine')
plt.axvline(x=120, color='green', linestyle='--')
plt.title('In the gif animations we show up to the point corresponding to the green line', color='green', fontsize=10)

plt.legend()
plt.savefig('animations/noise_schedule.png')
plt.show()

