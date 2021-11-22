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

img_path1 = 'dog.jpg'
img_path2 = 'example_2.JPEG'
img_path3 = 'perro_gato.jpg'
gt_category1 = 258 # samoyed
gt_category2 = 565  # freight car
gt_category3 = 281 # tabby cat

torch.manual_seed(0)
max_iterations = 301
l1_coeff = 1e-4
size = 224

model = models.googlenet(pretrained=True)
model.to(device)
#model = torch.nn.DataParallel(model, device_ids=[0,1])
model.eval()

init_time = time.time()

original_img_pil1 = Image.open(img_path1).convert('RGB')
original_img_pil2 = Image.open(img_path2).convert('RGB')
original_img_pil3 = Image.open(img_path3).convert('RGB')

# normalización de acuerdo al promedio y desviación std de Imagenet
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

img_normal1 = transform(original_img_pil1).unsqueeze(0)
img_normal2 = transform(original_img_pil2).unsqueeze(0)
img_normal3 = transform(original_img_pil3).unsqueeze(0)

img_batch = torch.cat((img_normal3, img_normal3, img_normal3))
img_batch.requires_grad = False
img_batch = img_batch.to(device)

org_softmax = torch.nn.Softmax(dim=1)(model(img_batch))  # tensor(3,1000)
prob_orig = org_softmax.data[[0,1,2],[gt_category3, gt_category3, gt_category3]].cpu().detach().numpy()
print(prob_orig)

for param in model.parameters():
    param.requires_grad = False

np.random.seed(seed=0)
mask = torch.from_numpy(np.float32(np.random.uniform(0, 0.01, size=(224, 224))))
mask = mask.expand(3, 1, 224, 224)
mask = mask.to(device)
mask.requires_grad = True

null_img = torch.zeros(3, 3, 224, 224).to(device)  # tensor (2, 3, 224, 224)

optimizer = torch.optim.Adam([mask], lr=0.1)

for i in range(max_iterations):
    extended_mask = mask
    extended_mask = extended_mask.expand(3, 3, 224, 224)
    perturbated_input = img_batch.mul(extended_mask) + null_img.mul(1 - extended_mask)
    #perturbated_input = perturbated_input.to(torch.float32)
    optimizer.zero_grad()
    outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  #(3,1000)

    preds = outputs[[0, 1, 2],[gt_category3, gt_category3, gt_category3]]

    loss = l1_coeff * torch.sum(torch.abs(1 - mask), dim=(1, 2, 3)) + preds
    loss.backward(gradient=torch.tensor([1., 1., 1.]).to(device))
    optimizer.step()
    #mask.data.clamp_(0, 1)

print('Time taken: {:.3f}'.format(time.time() - init_time))

mask_np = (mask.cpu().detach().numpy())
plt.imshow(1 - mask_np[0, 0, :, :])  # 1-mask para deletion
plt.show()
plt.imshow(1 - mask_np[1, 0, :, :])  # 1-mask para deletion
plt.show()
plt.imshow(1 - mask_np[2, 0, :, :])  # 1-mask para deletion
plt.show()
