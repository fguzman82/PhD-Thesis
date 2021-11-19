# version 1: DELETION WITHOUT REGU
import os
import cv2
import sys
import time
import scipy
import torch
import argparse
import numpy as np
import torch.optim
import shutil

from formal_utils import *
from skimage.transform import resize
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from torchvision import models

# sys.path.insert(0, './generativeimptorch')

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.googlenet(pretrained=True)
model.to(device)
model.eval()

init_time = time.time()

img = torch.randn(2, 3, 224, 224)
img.requires_grad = False
img = img.to(device)

org_softmax = torch.nn.Softmax(dim=1)(model(img))  # tensor(3,1000)
prob_orig = org_softmax.data[[0,1],[10,20]].cpu().detach().numpy()

for param in model.parameters():
    param.requires_grad = False

mask = torch.from_numpy(np.random.uniform(0, 0.01, size=(2, 1, 224, 224)))
mask = mask.to(device)
mask.requires_grad = True

null_img = torch.zeros(2, 3, 224, 224).to(device)  # tensor (1, 3, 224, 224)

optimizer = torch.optim.Adam([mask], lr=0.1)

for i in range(10):
    extended_mask = mask
    extended_mask = extended_mask.expand(2, 3, 224, 224)
    perturbated_input = img.mul(extended_mask) + null_img.mul(1 - extended_mask)
    perturbated_input = perturbated_input.to(torch.float32)
    optimizer.zero_grad()
    outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  #(3,1000)

    preds = outputs[[0,1],[10,20]]

    loss = 1e-4 * torch.sum(torch.abs(1 - mask), dim=(1, 2, 3)) + preds
    loss.backward(gradient=torch.tensor([1., 1.]).to(device))
    optimizer.step()
    mask.data.clamp_(0, 1)  


print('Time taken: {:.3f}'.format(time.time() - init_time))
