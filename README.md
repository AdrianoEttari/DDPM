# DDPM

In this repository you can find an application of a Deep Diffusion Model on the Animals-10 gold ranked dataset of kaggle https://www.kaggle.com/datasets/alessiocorrado99/animals10. Most of the code you can find here is based on the github repository in the following link https://github.com/dome272/Diffusion-Models-pytorch. I really suggest you to take a look at it, it is a very good repository and also he explaines everything in two well done youtube videos. I have just added some code to improve it:

**NOTICE: SOME OF THE FOLLOWING POINTS ARE STILL WORKING IN PROCESS**

* I wrote some functions in utils to get the FDI and the IS indices
* I changed the schedule from linear to cosine (actually I give the chance to the user to select the preferred one)
* I adapted the code for making it fault tolerant and working on multiple GPUs
* I added a new feature that consists in allowing the user to write something he wants to show up and then the algorithm generates images from this text

## Cosine vs Linear Schedule

![cosine schedule](https://github.com/AdrianoEttari/DDPM/blob/main/image_destruction_cos.gif)

![linear schedule](https://github.com/AdrianoEttari/DDPM/blob/main/image_destruction_linear.gif)
## Distributed Data Parallel (Or Distributed Training) and Fault Tolerance

Why we want to distribute our training? Because it is faster. If you have a machine with 4 GPUs, you can train your model 4 times faster than if you had only one GPU. 

When you work on a `single GPU`, during training, the algorithm takes an input batch from the Dataloader computes the forward pass, computes the loss, then computes the backward step and finally updates the parameters with the optimization step. 

When you work on `multiple GPUs`, you have to split the batch in multiple batches (the input batch is passed to the `DistributedSampler`), one for each GPU. Then, each GPU computes the forward pass, computes the loss, then computes the backward pass and because the inputs were different, the gradients that are accumulated are different. **Running the optimizer step on each GPU would result in different parameters and we would end up with 4 distinct models, instead with one single distributed model**. So, instead DDP runs a synchronization step after the backward pass, which means that the gradients are aggregated using the `Ring All Reduce algorithm`. The cool thing about this algorithm is that it doesn't wait for all the gradients to be computed before aggregating them. Instead it starts communication along the ring while the backward pass is still running (this ensures that your GPU is always working and not idle). Once the syncronization step is done, all the GPUs will have the same gradient and so we can run the optimizer step the aggregated gradients and the parameters are updated.
When you migrate from the single GPU to the multiple GPUs, you have to change the code. You have to assign a **rank** (unique identifier that is assigned to each process; usually ranges from 0 to world_size-1) and a **world_size** (total number of processors in the group). 

Pytorch provides fault tolerance with torchrun. Its idea is simple: your training script takes snapshots of your training job at regular intervals. Then, if a failure occurs your job doesn't exit, torchrun restarts all the processors which load the latest snapshot and you continue training from there. In this way you only lose the time between the last snapshot and the failure and you don't have to start from scratch. 

A snapshot includes the model parameters, the optimizer state, the epoch number, the iteration number, the loss, etc.

