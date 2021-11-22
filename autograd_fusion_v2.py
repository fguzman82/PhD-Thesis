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

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.float32(np.transpose(img.copy(), (2, 0, 1)))

    output_t = torch.from_numpy(output)
    if use_cuda:
        output_t = output_t.to('cuda')  # cuda()

    output_t.unsqueeze_(0)
    output_t.requires_grad = requires_grad
    return output_t

img_path1 = 'dog.jpg'
img_path2 = 'example_2.JPEG'
img_path3 = 'perro_gato.jpg'
gt_category1 = 258  # samoyed
gt_category2 = 565  # freight car
gt_category3 = 281  # tabby cat

#torch.manual_seed(0)
learning_rate = 0.3
max_iterations = 301
l1_coeff = 0.01e-5
size = 224

init_time = time.time()

torch.cuda.set_device(0)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
# model = torch.nn.DataParallel(model, device_ids=[0,1])
model.cuda()
model.eval()

list_of_layers = ['conv1',
                  'conv2',
                  'conv3',
                  'inception3a',
                  'inception3b',
                  'inception4a',
                  'inception4b',
                  'inception4c',
                  'inception4d',
                  'inception4e',
                  'inception5a',
                  'inception5b',
                  'fc'
                  ]
activation_orig = {}
gradients_orig = {}


def get_activation_orig(name):
    def hook(model, input, output):
        activation_orig[name] = output.clone()

    return hook


def get_gradients_orig(name):
    def hook(model, grad_input, grad_output):
        gradients_orig[name] = grad_output[0].cpu().detach().numpy()

    return hook


F_hook = []
#B_hook = []

for name, layer in model.named_children():
    if name in list_of_layers:
        F_hook.append(layer.register_forward_hook(get_activation_orig(name)))
        #B_hook.append(layer.register_backward_hook(get_gradients_orig(name)))

# original_img_pil1 = Image.open(img_path1).convert('RGB')
# original_img_pil2 = Image.open(img_path2).convert('RGB')
# original_img_pil3 = Image.open(img_path3).convert('RGB')

original_img_pil0 = Image.open('./dataset/0.JPEG').convert('RGB')
original_img_pil1 = Image.open('./dataset/1.JPEG').convert('RGB')
original_img_pil2 = Image.open('./dataset/2.JPEG').convert('RGB')
original_img_pil3 = Image.open('./dataset/3.JPEG').convert('RGB')
original_img_pil4 = Image.open('./dataset/4.JPEG').convert('RGB')
original_img_pil5 = Image.open('./dataset/5.JPEG').convert('RGB')
original_img_pil6 = Image.open('./dataset/6.JPEG').convert('RGB')
original_img_pil7 = Image.open('./dataset/7.JPEG').convert('RGB')
original_img_pil8 = Image.open('./dataset/8.JPEG').convert('RGB')
original_img_pil9 = Image.open('./dataset/9.JPEG').convert('RGB')

# normalización de acuerdo al promedio y desviación std de Imagenet
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# img_normal1 = transform(original_img_pil1).unsqueeze(0)
# img_normal2 = transform(original_img_pil2).unsqueeze(0)
# img_normal3 = transform(original_img_pil3).unsqueeze(0)

img_normal0 = transform(original_img_pil0).unsqueeze(0)
img_normal1 = transform(original_img_pil1).unsqueeze(0)
img_normal2 = transform(original_img_pil2).unsqueeze(0)
img_normal3 = transform(original_img_pil3).unsqueeze(0)
img_normal4 = transform(original_img_pil4).unsqueeze(0)
img_normal5 = transform(original_img_pil5).unsqueeze(0)
img_normal6 = transform(original_img_pil6).unsqueeze(0)
img_normal7 = transform(original_img_pil7).unsqueeze(0)
img_normal8 = transform(original_img_pil8).unsqueeze(0)
img_normal9 = transform(original_img_pil9).unsqueeze(0)

img_batch = torch.cat((img_normal0, img_normal1, img_normal2, img_normal3, img_normal4,
                       img_normal5, img_normal6, img_normal7, img_normal8, img_normal9
                       ))

print('tamaño del batch: ', img_batch.size(0))

img_batch.requires_grad = False
img_batch = img_batch.cuda()
org_softmax = torch.nn.Softmax(dim=1)(model(img_batch))  # tensor(3,1000)


#gt_category = [gt_category1, gt_category2, gt_category3]
gt_category = np.load('preds.npy').tolist()

# for i in range(img_batch.size(0)-3):
#     gt_category.append(gt_category3)

prob_orig = org_softmax.data[torch.arange(0, img_batch.size(0)).tolist(), gt_category].cpu().detach().numpy()

print(prob_orig)

for fh in F_hook:
    fh.remove()

# for bh in B_hook:
#     bh.remove()

#gradients = {}


def get_activation_mask(name):
    def hook(model, input, output):
        act_mask = output
        # print(act_mask.shape). #debug
        # print(activation_orig[name].shape) #debug
        limite_sup = (act_mask <= torch.fmax(torch.tensor(0), activation_orig[name]))
        limite_inf = (act_mask >= torch.fmin(torch.tensor(0), activation_orig[name]))
        oper = limite_sup * limite_inf
        # print('oper shape=',oper.shape). #debug
        act_mask.requires_grad_(True)
        act_mask.retain_grad()
        h = act_mask.register_hook(lambda grad: grad * oper)
        # x.register_hook(update_gradients(2))
        # activation[name]=act_mask
        # h.remove()

    return hook


# def get_act_mask_gradients(name):
#     def hook(model, grad_input, grad_output):
#         gradients[name] = grad_output[0]
#         # print('backward')
#         # return (new_grad,)
#
#     return hook


for name, layer in model.named_children():
    if name in list_of_layers:
        layer.register_forward_hook(get_activation_mask(name))
        #layer.register_backward_hook(get_act_mask_gradients(name))

for param in model.parameters():
    param.requires_grad = True

np.random.seed(seed=0)

mask = torch.from_numpy(np.random.uniform(0, 0.01, size=(1, 1, 224, 224)))
#mask = mask.expand(6, 1, 224, 224)
mask = mask.expand(img_batch.size(0), 1, 224, 224)
mask = mask.cuda()
mask.requires_grad = True

#null_img = torch.zeros(6, 3, 224, 224).to(device)  # tensor (2, 3, 224, 224)
#null_img = torch.zeros(img_batch.size(0), 3, 224, 224).cuda()
# imagen nulla difuminada
null_img_blur = transforms.GaussianBlur(kernel_size=223, sigma=10)(img_batch)
null_img_blur.requires_grad = False
null_img = null_img_blur.cuda()

optimizer = torch.optim.Adam([mask], lr=learning_rate)

for i in range(max_iterations):
    extended_mask = mask
    #extended_mask = extended_mask.expand(6, 3, 224, 224)
    extended_mask = extended_mask.expand(img_batch.size(0), 3, 224, 224)
    perturbated_input = img_batch.mul(extended_mask) + null_img.mul(1 - extended_mask)
    perturbated_input = perturbated_input.to(torch.float32)
    optimizer.zero_grad()
    outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # (3,1000)

    preds = outputs[torch.arange(0, img_batch.size(0)).tolist(), gt_category]

    loss = l1_coeff * torch.sum(torch.abs(1 - mask), dim=(1, 2, 3)) + preds
    #loss.backward(gradient=torch.tensor([1., 1., 1., 1., 1., 1.]).to(device))
    loss.backward(gradient=torch.ones_like(loss).cuda())
    #mask.grad.data = torch.nn.functional.normalize(mask.grad.data, p=float('inf'), dim=(2, 3))
    optimizer.step()
    mask.data.clamp_(0, 1)

print('Time taken: {:.3f}'.format(time.time() - init_time))

mask_np = (mask.cpu().detach().numpy())

for i in range(img_batch.size(0)):
    plt.imshow(1 - mask_np[i, 0, :, :])
    plt.show()

