import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta    # set the beta parameter for the exponential moving average
        self.step = 0       # step counter (initialized at 0) to track when to start updating the moving average

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()): #iterate over all parameters in the current and moving average models
            # get the old and new weights for the current and moving average models
            old_weight, up_weight = ma_params.data, current_params.data
            # update the moving average model parameter
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # if there is no old weight, return the new weight
        if old is None:
            return new
        # compute the weighted average of the old and new weights using the beta parameter
        return old * self.beta + (1 - self.beta) * new # beta is usually around 0.99
        # therefore the new weights influence the ma parameters only a little bit
        # (which prevents outliers to have a big effect) whereas the old weights
        # are more important.

    def step_ema(self, ema_model, model, step_start_ema=2000):
        '''
        We'll let the EMA update start just after a certain number of iterations
        (step_start_ema) to give the main model a quick warmup. During the warmup
        we'll just reset the EMA parameters to the main model one.
        After the warmup we'll then always update the weights by iterating over all
        parameters and apply the update_average function.
        '''
        # if we are still in the warmup phase, reset the moving average model to the current model
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        # otherwise update the moving average model parameters using the current model parameters
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        # reset the parameters of the moving average model to the current model parameters
        ema_model.load_state_dict(model.state_dict()) # we set the weights of ema_model
        # to the ones of model.



class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels       # store the number of channels
        self.size = size               # store the size of input image
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)   # Define a multi-head attention layer with 4 heads and set batch_first to True
        self.ln = nn.LayerNorm([channels]) # Remember that LayerNorm normalizes each object independently through
            # the object features (channels)
            
        # Define a feedforward neural network with two linear layers and a GELU activation function    
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Reshape the input tensor to have dimensions (-1, channels, size * size) and swap the 1st and 2nd dimensions
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) # .view() is like .reshape() in numpy. 
        # the -1 means that the first dimension will be inferred according the other two provided. For example, if you
        # have a tensor shaped (3,4) and you do tensor.view(-1, 3, 2) it will return a tensor shaped (2,3,2).
        # swapaxes() swaps the specified axes (in this case the 1st and the 2nd). 
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # The MultiheadAttention layer returns a tuple with a tensor and an Optional tensor (which is not needed).
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        '''
        We have just two convolutional layers, therefore we just need in_channels, mid_channels (what 
        the first convolution must return) and out_channels (what the second convolution must return).
        You can also not use the mid_channels and than the first convolution will return out_channels.

        There's also a little bit of ResNet in this class (in case residual=True)
        '''
        super().__init__()
        self.residual = residual # If True, the output of the double convolution will be the sum of the input and the output of the double convolution.
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # GroupNorm() performs the same formula of BatchNorm() but on a smaller group of channels (not on the whole batch).
            # This can help to reduce the memory and computational requirements of the normalization layer. So, it takes as input
            # the number of groups (images) to separate the channels into (remember that the array of an image batch is (batch_size, 
            # channels, height, width)) and the number of channels to normalize and it returns as output the normalized array of the
            # same shape. If num_groups = 1, then it is equivalent to having a batch size of batch_size=1 in nn.BatchNorm2d,
            # where each image is normalized independently.
            nn.GroupNorm(1, mid_channels),
            # GELU() (Gaussian Error Linear Unit) is a variant of the ReLU activation function. 
            # It is a smooth approximation of the ReLU function.
            # It is defined as: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) but actually 
            # the formula that you must remember is               GELU(x) = x * P(X <= x) 
            # where P(X <= x) is the CDF of the Gaussian (usually a standard one, but one could also set mu and sigma as learnable
            # parameters).
            # That's why usually if comes after a BatchNorm layer or a GroupNorm layer.
            # GELU is kind of a mix between ReLU and a Dropout layer because
            # for values of x particularly smaller than 0 (e.g. -2. Remember that in a standard Gaussian
            # the pdf for x<-2 is basically 0), GELU outputs 0 and so turn off the neuron. We can say that on the
            # extremes acts like a Relu (because if x>1 it approximate to x (because P(X<1)=1 and therefore
            # GELU(x>1)=x)). It solves the Dying ReLU problem and performs better than ReLU (indeed OpenAI used it in their GPT-3 model).
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        '''
        Performs a forward pass through the DoubleConv network with input x and returns
        the output of the network.
        '''
        if self.residual:
            #if residual=True, apply a residual connection between
            return F.gelu(x + self.double_conv(x)) # residual connection
        else:
            return self.double_conv(x)

class Down(nn.Module):
    '''
    The main part consists in reducing the size of the image by a factor of 2 (with a MaxPool2d layer) and then
    applying two Double convolutional layers. 

    The second part is the embedding layer. Since most blocks differ in terms of the hidden dimensions from the time 
    step embedding we'll make use of a linear projection to lead the time embedding in the proper dimension. 
    This consists of a SiLU activation and then a Linear projection which moves the time embedding from the emb_dim dimension
    to hidden dimensions.

    In the forward pass we first feed images to the convolutional block and project the time embedding accordingly. 
    Then we add both together and return.
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # downsample the image by a factor of 2
            # apply two double convolutional layers to the downsampled image
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),   # apply a SiLU activation function to the time embedding
            nn.Linear(emb_dim, out_channels),   # apply a linear projection to the time embedding to move it from emb_dim dimension to hidden dimensions

        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)   #feed the images to the convolutional block
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])   #project the time embedding
        return x + emb   #add the images and the projected time embedding and return the result

class Up(nn.Module):
    '''
    It's almost the same of the Down class but instead of having a MaxPool2d layer we have an Upsample layer.
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # concaatenate the skip connection with the upsampled image
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None):
        super().__init__()
        
        if os.environ.get("LOCAL_RANK")==None:
          self.gpu_id = 'cpu'
        else:
          self.gpu_id = int(os.environ["LOCAL_RANK"])
        
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim) # Here we create an embedding 
           # for the labels and for the time steps. num_classes in the embedding is the size
           # of the dictionary of the embedding and time_dim is the size of each embedding vector. 
           # So, the label embedding will have the same number of dimensions as the time embedding.

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            
            ** (torch.arange(0, channels, 2, device= torch.device('cpu' if self.gpu_id == 'cpu' else f'cuda:{self.gpu_id}')).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y) # In order to condition the model, 
            # we add the conditional information to some intermediate results. Luckily,
            # in diffusion models we already have the time step as condition, so, we
            # just need to add the label embedding (we don't use the labels in their plain forms)
            # to the time step embedding.

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
