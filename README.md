# DDPM

In this repository you can find an application of a Deep Diffusion Model on the Animals-10 gold ranked dataset of kaggle https://www.kaggle.com/datasets/alessiocorrado99/animals10. 

Regarding the organization of the material. In the `utils.py`, you can find some functions for plotting the images, for computing the FDI and the IS indices and also the function to get the dataloader. In the `modules.py` you can find all the classes regarding the UNet_conditional model. The `ddpm.py` is the main file, where all the other functions are called and where we implement all the functions regarding the deep diffusion models. `test.py` is a function where we performed some tests (you can skip it). Finally, in raw-img folder you find our images and in the animation folder you find some pictures and gifs to explain the theory and our results.

Most of the code you can find here is based on the github repository in the following link https://github.com/dome272/Diffusion-Models-pytorch. I really suggest you to take a look at it, it is a very good repository and also he explaines everything in two well done youtube videos. I have just added some code to improve it:

**NOTICE: SOME OF THE FOLLOWING POINTS ARE STILL WORKING IN PROCESS**

* I wrote some functions in utils to get the FDI and the IS indices
* I changed the schedule from linear to cosine (actually I give the chance to the user to select the preferred one)
* I adapted the code for making it fault tolerant and working on multiple GPUs
* I added a new feature that consists in allowing the user to write something he wants to show up and then the algorithm generates images from this text


## FDI and IS

Inception Score and Fréchet Distance (FDI) are two commonly used metrics in the field of deep learning to evaluate the performance of generative models.

### Inception Score (IS) **↑**

The Inception Score is a metric used to evaluate the quality of generated image**s. It measures how well a generative model can classify the images it generates**. **The score is calculated by feeding the generated images through a pre-trained inception network and calculating the entropy of the class predictions.**

A high Inception Score indicates that the generated images are diverse and visually appealing. However, it has been criticized for not accounting for the realism of the generated images, which can lead to artificially high scores.

However, when evaluating the IS on a true dataset, the expected score range is not the same as when evaluating it on a generated dataset. In general, the IS values for real images can range from 5 to 25, depending on the dataset and the specific evaluation method used **(notice that if the generated images are pretty realistic than the FDI will be small (positive thing) and since there exists the trade-off between FDI and IS, the IS will be small too (negative thing))**.

The Inception Score is calculated using the following formula:

$$
IS = e^{E[KL(p(y|x) || p(x))]}
$$

where p(y|x) is the class probability distribution of the generated images, p(x) is the class probability distribution of the real images, and KL is the Kullback-Leibler divergence.

### Fréchet Distance (FDI) **↓**

Fréchet Distance is another metric used to evaluate the quality of generative models. Unlike the Inception Score, **FDI measures the similarity between the distribution of real images and the distribution of generated images**. **It calculates the distance between the feature representations of the two distributions,** which are obtained by feeding them through a pre-trained convolutional neural network.

A low Fréchet Distance indicates that the generated images are realistic and similar to the real images. However, FDI is computationally more expensive than Inception Score and requires more data to be effective.

Both Inception Score and FDI are important tools for evaluating the quality of generative models. While Inception Score is good for evaluating the diversity of generated images, FDI is better for evaluating the realism of the generated images.

The Fréchet Distance is calculated using the following formula:

$$
FDI = ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2\sqrt{(sigma_1 * sigma_2)})
$$

where mu_1 and mu_2 are the mean feature representations of the real and generated images, and sigma_1 and sigma_2 are the covariance matrices of the real and generated images, respectively.

In diffusion models (with CFG) you get lower FDI with a small amount of guidance (w=0.1 or w=0.3).
## Cosine vs Linear Schedule
As explained in the OpenAI paper (https://arxiv.org/pdf/2102.09672.pdf) in the chapter 3.2, the linear schedule works well for high resolution images, but for 64x64 or 32x32 images it's sub-optimal. The main problem is that the linear schedule is redundant at the last steps (it converges to random noise too fast), indeed if you skip the last 20% of the noise steps, the results quality doesn't change much. The cosine schedule solves this problem. As suggested by the authors, every function that acts like the cosine schedule that you see in the following plot is a good candidate for the noise schedule.

![image_destruction_cosine_120](https://user-images.githubusercontent.com/120527637/221939925-9e6331f4-4c0a-4df3-8b5b-b849f0a59cfc.gif)
![image_destruction_linear_120](https://user-images.githubusercontent.com/120527637/221939950-69e41fe4-f008-4244-a522-0f06f19ea5f6.gif)

![noise_schedule](https://user-images.githubusercontent.com/120527637/221933361-8352564b-db20-4942-9f1a-dfbad2acb3d3.png)


## Distributed Data Parallel (Or Distributed Training) and Fault Tolerance

Why we want to distribute our training? Because it is faster. If you have a machine with 4 GPUs, you can train your model 4 times faster than if you had only one GPU. 

When you work on a `single GPU`, during training, the algorithm takes an input batch from the Dataloader computes the forward pass, computes the loss, then computes the backward step and finally updates the parameters with the optimization step. 

When you work on `multiple GPUs`, you have to split the batch in multiple batches (the input batch is passed to the `DistributedSampler`), one for each GPU. Then, each GPU computes the forward pass, computes the loss, then computes the backward pass and because the inputs were different, the gradients that are accumulated are different. **Running the optimizer step on each GPU would result in different parameters and we would end up with 4 distinct models, instead with one single distributed model**. So, instead DDP runs a synchronization step after the backward pass, which means that the gradients are aggregated using the `Ring All Reduce algorithm`. The cool thing about this algorithm is that it doesn't wait for all the gradients to be computed before aggregating them. Instead it starts communication along the ring while the backward pass is still running (this ensures that your GPU is always working and not idle). Once the syncronization step is done, all the GPUs will have the same gradient and so we can run the optimizer step the aggregated gradients and the parameters are updated.
When you migrate from the single GPU to the multiple GPUs, you have to change the code. You have to assign a **rank** (unique identifier that is assigned to each process; usually ranges from 0 to world_size-1) and a **world_size** (total number of processors in the group). 

Pytorch provides fault tolerance with torchrun. Its idea is simple: your training script takes snapshots of your training job at regular intervals. Then, if a failure occurs your job doesn't exit, torchrun restarts all the processors which load the latest snapshot and you continue training from there. In this way you only lose the time between the last snapshot and the failure and you don't have to start from scratch. 

A snapshot includes the model parameters, the optimizer state, the epoch number, the iteration number, the loss, etc.

