import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

def plot_images(images):
    plt.figure(figsize=(12, 12))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu()) # we need to permute the tensor because the default of torch
    # is to have the channels in the first dimension, but matplotlib expects the channels
    # in the last dimension. Also we need to convert the tensor with .cpu() to a numpy array
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    ndarr = (255 * ndarr).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(path)
    # The ndarr variable is created by converting the PyTorch tensor to a numpy array with the
    # .to('cpu').numpy() method. By default, this method returns a numpy array with the float32 
    # data type, which is not supported by the Image.fromarray() function.
    # im = Image.fromarray(ndarr)
    # im.save(path)

def get_data_ddp(image_size, dataset_path: str, batch_size: int):
    '''
    This function transforms the images in the dataset_path folder and returns a dataloader.
    A DistributedSampler is used to split the batch inputs among the different processes.

    Input:
        image_size: the size of the images that will be returned by the dataloader
        dataset_path: the path to the folder containing the images
        batch_size: the size of the batches that will be returned by the dataloader

    Output:
        dataloader: a dataloader that returns batches of images
    '''
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80), 
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalizes 
        # each of the three channels of the images with a mean of 0.5 and a standard deviation of 0.5
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms) # in datasets_path there are subfolders with images
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset))
    return dataloader

def IS_computer(generated_data_path, splits):
    '''
    This function computes the Inception Score of the images in the generated_data_path folder.

    Input:
        generated_data_path: the path to the folder containing the generated images
        splits: the number of splits to use to compute the Inception Score

    Output:
        is_mean: the mean of the Inception Score
        is_std: the standard deviation of the Inception Score
    '''
    transform = torchvision.transforms.Compose([
         torchvision.transforms.Resize(299),   # resize the image to 299x299 pixels
         torchvision.transforms.CenterCrop(299),  # crop the image to 299x299 pixels at the center
         torchvision.transforms.ToTensor(),   # convert the image to a tensor
         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

    dataset = torchvision.datasets.ImageFolder(generated_data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = torchvision.models.inception_v3(pretrained=True, transfrom_input=False).cuda().eval()

    preds = []
    for image, label in dataloader:
        with torch.no_grad():
            pred = model(image.cuda())
            pred = torch.nn.functional.F.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)

        scores = []
        splits = splits
        for i in range(splits):
                part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
                kl = np.mean(np.sum(kl, axis=1))
                scores.append(np.exp(kl))
    is_mean, is_std = np.mean(scores), np.std(scores)
    return is_mean, is_std




def FDI_computer(real_images, generated_images):
    '''
    This function computes the Fr??chet Distance using the real_images and generated_images.

    Input:
        real_images: a tensor containing the real images
        generated_images: a tensor containing the generated images

    Output:
        FDI: the Fr??chet Distance between the real and generated images
    '''
    from scipy.linalg import sqrtm
    model = torchvision.models.inception_v3(pretrained=True, transfrom_input=False).cuda().eval()
    # Generate feature representations of real and generated images
    real_features = model.predict(real_images)
    generated_features = model.predict(generated_images)
    # Calculate mean and covariance of feature representations
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    # Calculate the Fr??chet Distance
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # YOU CAN ALSO USE THE FOLLOWING PYTORCH CODE INSTEAD OF USING SCIPY

    # covmean = torch.mm(torch.sqrt(sigma1), torch.sqrt(sigma2)) # THIS IS ONE WAY

    # eigvals, eigvecs = torch.linalg.eig(A) # THIS IS THE OTHER WAY
    # eigvals_sqrt = torch.diag(torch.sqrt(eigvals))
    # covmean = eigvecs @ eigvals_sqrt @ torch.linalg.inv(eigvecs)

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid