import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from models import UNet_conditional, EMA
import numpy as np
import copy
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import imageio

def ddp_setup():
    '''
    This function is used to setup the distributed data parallelism (DDP) in Pytorch.
    It is used to train the model on multiple GPUs.

    init_process_group() is used to initialize the process group. It is used to synchronize the processes
    and to make sure that the processes are using the same random seed. This is important because we want
    to make sure that the same random noise is added to the images in each GPU. If we don't do this, the
    images will be different in each GPU and the model will not be able to learn anything.
    '''
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
        
        if os.environ.get("LOCAL_RANK")==None:
            self.gpu_id = 'cpu'
        else:
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

        if self.prepare_noise_schedule == 'linear':
            self.beta = self.prepare_noise_schedule(noise_schedule=self.noise_schedule).to(self.gpu_id) 
        # The reason why we use the method 'to' is that we want to move the tensor to the gpu_id we specified in the constructor
        # (by default Pytorch stores it in the default memory location, which is usually the CPU).
        # When you want to perform computations with the tensor on another gpu_id, such as a GPU, you need to move the tensor to
        # the corresponding gpu_id memory using the 'to' method.
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Notice that beta is not just a number, it is a tensor of shape (noise_steps,).
        # If we are in the step t then we index the tensor with t. To get alpha_hat we compute the cumulative product of the tensor.

        elif self.prepare_noise_schedule == 'cosine':
            self.alpha_hat = self.prepare_noise_schedule(noise_schedule=self.noise_schedule).to(self.gpu_id)


    def prepare_noise_schedule(self, noise_schedule):
        '''
        In this function we set the noise schedule to use. Basically, we need to know how much gaussian noise we want to add
        for each noise step.

        Input:
            noise_schedule: the name of the noise schedule to use. It can be 'linear' or 'cosine'.

        Output:
            if noise_schedule == 'linear':
                self.beta: a tensor of shape (noise_steps,) that contains the beta values for each noise step.
            elif noise_schedule == 'cosine':
                self.alpha_hat: a tensor of shape (noise_steps,) that contains the alpha_hat values for each noise step.
        '''
        if noise_schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif noise_schedule == 'cosine':
            f_t = torch.cos(((((torch.arange(self.noise_steps)/self.noise_steps)+0.008)/(1+0.008))*torch.pi/2))**2 # Here we apply the formula of the OpenAI paper https://arxiv.org/pdf/2102.09672.pdf
            alpha_hat = f_t/f_t[0]  
            return alpha_hat

    def noise_images(self, x, t):
        '''
        ATTENTION: The error Ɛ is random, but how much of it we add to move forward depends on the Beta schedule.

        Input:
            x: the image at the previous timestep (x_{t-1})
            t: the current timestep
        
        Output:
            x_t: the image at the current timestep (x_t)
            Ɛ: the error that we add to x_t to move forward
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # Each None is a new dimension (e.g.
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)). It doens't add
        # them exatly in the same place, but it adds them in the place where the None is.
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x) # torch.randn_like() returns a tensor of the same shape of x with random values from a standard gaussian (notice that the values inside x are not relevant)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        '''
        During the training we sample t from a Uniform discrete distribution (from 1 to T)

        For each image that I have in the training, I want to sample a timestep t from a uniform distribution
        (notice that it is not the same for each image). 

        Input:
            n: the number of images we want to sample the timesteps for

        Output:
            t: a tensor of shape (n,) that contains the timesteps for each image
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward (moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY).

        What we do is to predict the noise conditionally, then if the cfg_scale is > 0,
        we also predict the noise unconditionally. Eventually, we apply the formula
        out of the CFG paper using the torch.lerp function which does exactly the same

        predicted_noise = uncond_predicted_noise + cfg_scale * (predicted_noise - uncond_predicted_noise)

        and this will be our final predicted noise.

        Input:
            model: the model that predicts the gaussian noise of an image
            n: the number of images we want to sample
            labels: the labels of the images we want to sample
            cfg_scale: the scale of the CFG noise
        
        Output:
            x: a tensor of shape (n, 3, self.img_size, self.img_size) with the generated images
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
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale) # Compute the formula of the CFG paper https://arxiv.org/abs/2207.12598
                
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
        '''
        This function doesn't output anything, but it saves the model state, the optimizer state and the current epoch.
        It is a mandatory function in order to be fault tolerant.

        Input:
            epoch: the current epoch

        Output:
            None
        '''
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path):
        '''
        This function doesn't return anything. It loads the model state, the optimizer state and the current epoch from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.
        
        Input:
            snapshot_path: the path of the snapshot

        Output:
            None
        '''
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train(self, lr, image_size, epochs):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            image_size: the size of the images
            epochs: the number of epochs
        '''
        model = self.model
        dataloader = self.train_data
        gpu_id = self.gpu_id
        noise_schedule = self.noise_schedule

        optimizer = optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.
        mse = nn.MSELoss()
        diffusion = Diffusion(img_size=image_size, gpu_id=gpu_id, noise_schedule=noise_schedule) # create the diffusion object

        l = len(dataloader) # number of batches in the dataloader
        ema = EMA(beta = 0.995) # create the EMA object
        ema_model = copy.deepcopy(model).eval().requires_grad_(False) # create the copy of the model
        # Remember that EMA works by creating a copy of the initial model weights and then
        # update them with moving average for the main model (w = B * w_{old} + (1-B) * w_{new})

        for epoch in range(epochs):
            pbar = tqdm(dataloader)
            for i, (images, labels) in enumerate(pbar):
                images = images.to(gpu_id) # move the images to the gpu
                labels = labels.to(gpu_id) # move the labels to the gpu
                t = diffusion.sample_timesteps(images.shape[0]).to(gpu_id)
                # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)
                # with random integers from 1 to noise_steps.
                x_t, noise = diffusion.noise_images(images, t) # here I'm going to get batch_size noise images
                if np.random.random() < 0.1: # 10% of the time, don't pass labels (we train 10% 
                    # of the times uncoditionally and 90% conditionally)
                    labels = None
                predicted_noise = model(x_t, t, labels) # here we pass the plain labels to the model (i.e. 0,1,2,...,9 if there are 10 classes)
                # The labels are created according to the order of the class folders (the first folder will have class 0).
                loss = mse(noise, predicted_noise) # compute the loss

                optimizer.zero_grad() # set the gradients to 0
                loss.backward() # compute the gradients
                optimizer.step() # update the weights
                ema.step_ema(ema_model, model) # call the step_ema after every model update

                pbar.set_postfix(MSE=loss.item()) # set_postfix just adds a message or value
                # displayed after the progress bar. In this case the loss of Batch i.

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                # labels = torch.arange(args.num_classes).long().to(gpu_id)
                # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

    def gif_forward_creator(self, image_path, save_path, fps=24):
        '''
        This function creates a gif of the diffusion process. It takes an image, adds noise for each 
        timestep and then all the noise images are concatenated and saved as a gif.

        Input:
            image_path: the path of the image
            save_path: the path where the gif will be saved
            fps: the number of frames per second

        Output: 
            None
        '''
        alpha_hat = self.alpha_hat 
        img_from_data = Image.open(image_path)
        image = torchvision.transforms.ToTensor()(img_from_data)[None, :] # add a batch dimension
        t = torch.arange(0, self.noise_steps).long() # create a tensor with the timesteps
        noised_image, _ = self.noise_images(image, t, alpha_hat) # add noise to the image
        images = torch.cat([image, noised_image], dim=0) # concatenate the images
        pil_images = [torchvision.transforms.ToPILImage()(images[i]) for i in range(self.noise_steps+1)] # convert the images to PIL images
        imageio.mimsave(save_path, pil_images, fps=fps) # save the gif


def launch(num_classes: int,
            image_size,
            dataset_path: str,
            batch_size: int,
            lr: float,
            epochs: int,
            noise_schedule: str,
            save_every: int,
            snapshot_path: str,
            output_path: str):
    '''
    Don't get confused by image_size and img_size. The first is the size of the images in the dataset and will be passed to the dataloader.
    The second is passed just to the sample to generate images. You can tweak both of them.

    This function is the main. You just need to run it in order to train the model.

    Input:
        num_classes: the number of classes
        image_size: the size of the images in the dataset
        dataset_path: the path of the dataset
        batch_size: the batch size
        lr: the learning rate
        epochs: the number of epochs
        noise_schedule: the noise schedule (linear, cosine)
        save_every: the number of epochs after which the model will be saved
        snapshot_path: the path where the model will be saved
        output_path: the path where the output will be saved

    Output:
        None
    '''
    ddp_setup() # this function is used to setup the distributed data parallel
    path = output_path 
    os.makedirs(path, exist_ok=True) # create the output path if it doesn't exist
    dataloader = get_data_ddp(image_size, dataset_path, batch_size) # create the dataloader
    model = UNet_conditional(num_classes=num_classes) # create the model
    diffusion = Diffusion(noise_schedule=noise_schedule, save_every=save_every, model=model, 
                          train_data=dataloader, snapshot_path=os.path.join(path, snapshot_path),
                          noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                          img_size=256) # create the diffusion object (Notice that 
    # img_size is different from image_size, as specified in the docstring)
    diffusion.train(lr, image_size, epochs) # train the model
    destroy_process_group() # destroy the process group that was initialized by ddp_setup()
    # When the destroy_process_group() function is called, all resources associated with
    # the process group are freed, including network connections and allocated memory. 

if __name__ == '__main__':
    import argparse     
    parser = argparse.ArgumentParser(description='DDPM conditional with EMA and cosine schedule')
    parser.add_argument('--dataset_path', type=str, default='raw-img')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--snapshot_path', type=str, default='snapshot.pt') # You just need to pass the name of the snapshot file.
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    launch(args.dataset_path, args.epochs, args.batch_size, args.image_size, args.num_classes, args.lr, args.save_every, args.noise_schedule, args.snapshot_path, args.output_path)
    

