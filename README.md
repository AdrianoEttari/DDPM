# DDPM

In this repository you can find an application of a Deep Diffusion Model on the Animals-10 gold ranked dataset of kaggle https://www.kaggle.com/datasets/alessiocorrado99/animals10. Most of the code you can find here is based on the github repository in the following link https://github.com/dome272/Diffusion-Models-pytorch. I really suggest you to take a look at it, it is a very good repository and also he explaines everything in two well done youtube videos. I have just added some code to improve it:
*  I wrote some functions in utils to get the FDI and the IS indices
* I changed the schedule from linear to cosine
* I added some snippets for making it fault tolerant and working on multiple GPUs
* I added a new feature that consists in allowing the user to write something he wants to show up and then the algorithm generates images from this text
* Eventually, I added some code to make it working on the Animal-10 dataset. 

