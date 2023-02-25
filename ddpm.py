import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import numpy as np
import copy
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend="nccl") # nccl is a collective communication library that is optimized for NVIDIA GPUs

# We will use the same number of noise steps for training and for sampling. It is not mandatory and we will do it just
# to be practical, but notice that it is common to use a larger number of sampling steps during inference
# than during training. This is because a larger number of sampling steps can lead to more accurate and diverse samples.
class Diffusion:
    def __init__(
            self,
            noise_schedule: str,
            save_every: int,
            model: nn.Module,
            train_data,
            snapshot_path: str,
            noise_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            img_size=256):
    
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.gpu_id = int(os.environ["LOCAL_RANK"])

        self.train_data = train_data
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.save_every = save_every
        self.epochs_run = 0
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, gpu_id_ids=[self.gpu_id])
        self.noise_schedule = noise_schedule

        self.beta = self.prepare_noise_schedule(noise_schedule=self.noise_schedule).to(self.gpu_id) # The reason why we use the method 'to' is that
        # we want to move the tensor to the gpu_id we specified in the constructor (by default Pytorch 
        # stores it in the default memory location, which is usually the CPU).When you want to perform computations
        # with the tensor on another gpu_id, such as a GPU, you need to move the tensor to the corresponding
        # gpu_id memory using the 'to' method.
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # currently (10/02/23) torch.cumprod() doesn't work with mps gpu_id. If you are using cuda, you can use it.
        # alpha_cpu= self.alpha.cpu()
        # result = np.cumprod(alpha_cpu.numpy(), axis=0) # Notice that beta is not just a number, it is a tensor of shape (noise_steps,). If we are in the step t then we index the tensor with t. To get alpha_hat we compute the cumulative product of the tensor.
        # self.alpha_hat = torch.from_numpy(result).to(gpu_id)


    def prepare_noise_schedule(self, noise_schedule):
        if noise_schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif noise_schedule == 'cosine':
            pass

    def noise_images(self, x, t):
        '''
        This function returns the x_t noise image (check the training algorithm)

        ATTENTION: The error Ɛ is random, but how much of it we add to move forward, depends on the Beta schedule.
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # Each None is a new dimension (e.g.
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)). It doens't add
        # them exatly in the same place, but it adds them in the place where the None is.
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x) # torch.randn_like() returns a tensor of the same shape of x with random values from a standard gaussian (notice that the values inside x are not relevant)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        '''
        During the training we sample t from a Uniform distribution (from 1 to T)

        For each image that I have in the training, I want to sample a timestep t from a uniform distribution
        (notice that it is not the same for each image). 
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward (moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY).

        This function takes the model and the number of images we want to sample (it considers n full noise images) 
        and returns (out of the noise) a tensor of shape (n, 3, self.img_size, self.img_size)
        with the generated images.

        What we do is to predict the noise conditionally, then if the cfg_scale is > 0,
        we also predict the noise unconditionally. Eventually, we apply the formula
        out of the CFG paper using the torch.lerp function which does exactly the same

        predicted_noise = uncond_predicted_noise + cfg_scale * (predicted_noise - uncond_predicted_noise)

        and this will be our final predicted noise.
        '''

        model.eval() # disables dropout and batch normalization
        with torch.no_grad(): # disables gradient calculation
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.gpu_id) # generates n noisy images of shape (3, self.img_size, self.img_size)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # tqdm just makes the loop show a progress bar.
                # To be precise, it shows something like 999999it [00:04, 247411.64it/s]. Where 999999it is the
                # number of iterations, 00:04 is the time elapsed to compute these iterations. 247411.64it/s is the
                # number of iterations per second. The position=0 is just to make the progress bar appear in the
                # first line of the terminal (position=1 -> the progress bar is displayed on the second line of the terminal).
                t = (torch.ones(n) * i).long().to(self.gpu_id) # tensor of shape (n) with all the elements equal to i.
                # Basically, each of the n image will be processed with the same integer time step t.
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    # If i>1 then we add noise to the image we have sampled (remember that from x_t we sample x_{t-1}).
                    # If i==1 we sample x_0, which is the final image we want to generate, so we don't add noise.
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) # we don't add noise (it's equal to 0) in the last time step because it would just make the final outcome worse.
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train() # enables dropout and batch normalization
        x = (x.clamp(-1, 1) + 1) / 2 # clamp takes a minimum and a maximum. All the terms that you pass
        # as input to it are than modified: if their are less than the minimum, clamp outputs the minimum, 
        # otherwise outputs them. The same (but opposit reasoning) for the maximum.
        # +1 and /2 just to bring the values back to 0 to 1.
        x = (x * 255).type(torch.uint8)
        return x

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train(self, lr, image_size, epochs):
        model = self.model
        dataloader = self.train_data
        gpu_id = self.gpu_id
        noise_schedule = self.noise_schedule

        optimizer = optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.
        mse = nn.MSELoss()
        diffusion = Diffusion(img_size=image_size, gpu_id=gpu_id, noise_schedule=noise_schedule)

        l = len(dataloader)
        ema = EMA(beta = 0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False) # create the copy of the model
        # Remember that EMA works by creating a copy of the initial model weights and then
        # update them with moving average for the main model (w = B * w_{old} + (1-B) * w_{new})

        for epoch in range(epochs):
            pbar = tqdm(dataloader)
            for i, (images, labels) in enumerate(pbar):
                images = images.to(gpu_id)
                labels = labels.to(gpu_id)
                t = diffusion.sample_timesteps(images.shape[0]).to(gpu_id)
                # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)
                # with random integers from 1 to noise_steps.
                x_t, noise = diffusion.noise_images(images, t) # here I'm going to get batch_size noise images
                if np.random.random() < 0.1: # 10% of the time, don't pass labels (we train 10% 
                    # of the times uncoditionally and 90% conditionally)
                    labels = None
                predicted_noise = model(x_t, t, labels) # here we pass the plain labels to the model (i.e. 0,1,2,...,9 if there are 10 classes)
                # The labels are created according to the order of the class folders (the first folder will have class 0).
                loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model) # call the step_ema after every model update

                pbar.set_postfix(MSE=loss.item()) # set_postfix just adds a message or value
                # displayed after the progress bar. In this case the loss of Batch i.

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                # labels = torch.arange(args.num_classes).long().to(gpu_id)
                # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)


def launch(num_classes: int,
            image_size,
            dataset_path: str,
            batch_size: int,
            lr: float,
            epochs: int,
            noise_schedule: str,
            save_every: int,
            snapshot_path: str):
    '''
    Don't get confused by image_size and img_size. The first is the size of the images in the dataset and will be passed to the dataloader.
    The second is passed just to the sample to generate images. You can tweak both of them.
    '''
    ddp_setup()
    os.makedirs('weights', exist_ok=True)
    dataloader = get_data_ddp(image_size, dataset_path, batch_size)
    model = UNet_conditional(num_classes=num_classes)
    diffusion = Diffusion(noise_schedule=noise_schedule, save_every=save_every, model=model, 
                          train_data=dataloader, snapshot_path=snapshot_path,
                          noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                          img_size=256)
    diffusion.train(lr, image_size, epochs)
    destroy_process_group()

if __name__ == '__main__':
    import argparse     
    parser = argparse.ArgumentParser(description='DDPM conditional with EMA and cosine schedule')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--snapshot_path', type=str, default='weights/snapshot.pt')
    args = parser.parse_args()
    launch(args.dataset_path, args.epochs, args.batch_size, args.image_size, args.num_classes, args.lr, args.save_every, args.noise_schedule, args.snapshot_path)
    # torchrun --standalone --nproc_per_node=cpu ddpm.py --dataset_path=raw-img/
