import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import warnings, sys

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    filesize = os.path.getsize(filename)
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()

def show_images(images, ncol=12, figsize=(8,2), title=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    out = rearrange(images, '(b1 b2) c h w -> c (b1 h) (b2 w)', b2=ncol).cpu()
    if title is not None:
        fig.suptitle(title)
    if out.shape[0] == 1:
        ax.matshow(out[0], **kwargs)
    else:
        ax.imshow(out.permute((1, 2, 0)), **kwargs)
    display.display(fig)
    plt.close(fig)