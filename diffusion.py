import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """
    def __init__(self, model, timesteps = 1000, 
                 beta_start = 1e-4, beta_end = 0.02):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.inference_steps = torch.arange(self.timesteps-1, -1, -1)
        
        # define beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        betas_sqrt = self.betas.sqrt()
        # dict for legibility and fast access
        self.b = {'b': self.betas, 
                  'sqrt': betas_sqrt}

        # define the alpha variable for ease of notation and to save on computations
        self.alphas = 1.0 - self.betas
        alphas_sqrt = self.alphas.sqrt()
        alpha_bars = self.alphas.cumprod(dim=-1)
        alpha_bars_sqrt = alpha_bars.sqrt()
        alpha_bars_sqrt_minus_one = (1-alpha_bars).sqrt()
        #dict for legibility and fast access
        self.a = {'a': self.alphas,
                  'sqrt': alphas_sqrt,
                  'bar': alpha_bars,
                  'bar sqrt': alpha_bars_sqrt,
                  '1-sqrt': alpha_bars_sqrt_minus_one}

    
    def forward(self, images, t, noise = None):
        """Forward diffusion process: q(x_t | x_0)
        Args:
        images of shape (batch_size, n_channels, H, W): Conditioning images.
        t of shape (batch_size, ): Corruption temperatures
        noise of shape (batch_size, n_channels, H, W): Noise added to the images, None for no noise
        
        Returns:
        x of shape (batch_size, n_channels, H, W): Generated samples (one sample per input image).
        """
        if noise is None:
            noise = torch.zeros_like(x_0)

        a0 = self.a['bar sqrt'][t][:,None,None,None] # square roots of alpha bars
        a1 = self.a['1-sqrt'][t][:,None,None,None]   # one minus alpha bars square roots
        x = a0 * images + a1 * noise
        
        return x
    
    def loss(self, images, t, labels = None, noise = None):
        """Calculate training loss via MSE
        Args:
        images of shape (batch_size, n_channels, H, W): Training images
        t of shape (batch_size,): Selected temperatures
        labels of shape (batch_size,): Classes of images, None if no conditioning of classes
        noise of shape (batch_size, n_channels, H, W): Noise we try to predict, standard gaussian by default

        Returns:
        loss (float): MSE loss of the predicted noise vs generated noise
        """
        if noise is None:
            noise = torch.randn_like(images)
        
        x_t = self.forward(images, t, noise)
        predicted_noise = self.model(x_t, t, labels)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        return loss


    @torch.no_grad()
    def inpaint(self, images, mask_known, labels=None):
        """Generate samples conditioned on known parts of images.
        
        Args:
        images of shape (batch_size, n_channels, H, W): Conditioning images.
        mask_known of shape (batch_size, 1, H, W): BoolTensor which specifies known pixels in images (marked as True).
        labels of shape (batch_size,): Classes of images, None for no conditioning on classes.
        
        Returns:
        x of shape (batch_size, n_channels, H, W): Generated samples (one sample per input image).
        """
        noise = torch.randn_like(images)
        x_t = noise
        t_tensor = torch.ones(images.size(0))

        # clean up the loop by omitting the self
        a = self.a
        b = self.b
        for t in tqdm(reversed(range(self.timesteps))):
            a_t = a['a'][t]
            b_t = b['b'][t]

            squiggly_b = (1-a['bar'][t-1])/(1-a['bar'][t])*b_t

            temps = t_tensor * t
            z = torch.randn_like(images)

            x_mask = a['bar sqrt'][t-1] * b_t / (1-a['bar'][t]) * images[mask_known] + a['sqrt'][t] * (1-a['bar'][t-1]) / (1-a['bar'][t]) * x_t[mask_known]
            x_mask += squiggly_b.sqrt() * z[mask_known]

            pred_noise = self.model(x_t, temps, labels)

            x_t = 1/a['sqrt'][t] * (x_t - b_t / a['1-sqrt'][t] * pred_noise ) + b['sqrt'][t] * z 
            x_t[mask_known] = x_mask

        x0 = x_t
        x0[mask_known] = images[mask_known]
            
        return x0
    
    @torch.no_grad()
    def sample(self, x_shape, labels = None, init_x = None):
        """Generate samples conditioned on labels
        Args:
        x_shape: Tuple containing the dimensions of the output
        labels of shape (batch_size,): Labels to be conditioned on
        init_x of shape (batch_size, n_channels, H, W): Starting point of the diffusion, None for a random start

        Returns:
        x_0 of shape x_shape: Generated samples
        """
        # start from pure noise if no initial x
        if init_x is not None:
            x_t = init_x
        else:
            x_t = torch.randn(x_shape)

        # multiply this tensor by t to get the temp tensor
        t_tensor = torch.ones(x_shape[0])

        # clean the loop by omitting the self        
        a = self.a
        b = self.b

        # iteratively denoise
        for t in tqdm(reversed(range(self.timesteps-1))):
            a_t = self.a['a'][t]
            b_t = self.b['b'][t]

            temps = t_tensor * t
            z = torch.randn(x_shape)

            pred_noise = self.model(x_t, temps, labels)

            x_t = 1 / a['sqrt'][t] * (x_t - b_t / a['1-sqrt'][t] * pred_noise) + b['sqrt'][t] * z
        temps = t_tensor * 0
        a0 = a['a'][0]
        b0 = b['b'][0]

        x_0 = 1 / a0 * (x_t - b0 / a['1-sqrt'][0] * model(x_t, temps, labels))
                                      
        return x_0