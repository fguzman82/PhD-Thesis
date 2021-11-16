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
import torchvision.transforms as transforms
import torchvision.utils as vutils

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# bibliotecas inpainter
sys.path.insert(0, './generativeimptorch')
from utils.tools import get_config, get_model_list
from model.networks import Generator


def inpainter(img, mask):
    config = get_config('./generativeimptorch/configs/config.yaml')
    checkpoint_path = os.path.join('./generativeimptorch/checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    cuda = config['cuda']
    device_ids = config['gpu_ids']

    with torch.no_grad():  # enter no grad context
        # Test a single masked image with a given mask
        x = img
        # denormaliza imagenet y se normaliza a inpainter [-1,1] mean=0.5, std=0.5
        x = transforms.Normalize(mean=[0.015 / 0.229, 0.044 / 0.224, 0.094 / 0.225],
                                 std=[0.5 / 0.229, 0.5 / 0.224, 0.5 / 0.225])(x)
        x = x * (1. - mask)
        # Define the trainer
        netG = Generator(config['netG'], cuda, device_ids)
        # Resume weight
        last_model_name = get_model_list(checkpoint_path, "gen", iteration=0)
        netG.load_state_dict(torch.load(last_model_name))
        model_iteration = int(last_model_name[-11:-3])
        print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

        if cuda:
            netG = torch.nn.parallel.DataParallel(netG, device_ids=[0, 1])
            netG.cuda()
            x = x.cuda()
            mask = mask.cuda()

            # Inference
            x1, x2, offset_flow = netG(x, mask)
            inpainted_result = x2 * mask + x * (1. - mask)
    return inpainted_result


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


def numpy_to_torch2(img):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    output.unsqueeze_(0)
    return output


if __name__ == '__main__':

    # img_path = 'perro_gato.jpg'
    # img_path = 'dog.jpg'
    # img_path = 'example.JPEG'
    img_path = 'example_2.JPEG'

    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_img_pil = Image.open(img_path).convert('RGB')
    original_np = np.array(original_img_pil)

    # normalización de acuerdo al promedio y desviación std de Imagenet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # se normaliza la imágen y se agrega una dimensión [1,3,244,244]
    img_normal = transform(original_img_pil).unsqueeze(0)  # Tensor (1, 3, 224, 224)
    img_normal.requires_grad = False

    img = img_normal  # tensor (1, 3, 224, 224)

    mask = torch.zeros(1, 1, 224, 224)
    mask[:, :, 100:130, 120:200] = 1.0

    img_inpainted = inpainter(img, mask)
    vutils.save_image(img_inpainted, 'output2.png', padding=0, normalize=True)

    #denormalizado
    img_inpainted = transforms.Normalize(mean=-1, std=2)(img_inpainted)


    img_normal_np = img_inpainted.cpu().detach().numpy()
    img_transform_T = np.moveaxis(img_normal_np[0, :].transpose(), 0, 1)
    plt.imshow(img_transform_T)
    plt.show()

    mask_T = np.moveaxis(mask.numpy()[0, :].transpose(), 0, 1)
    plt.imshow(mask_T)
    plt.show()
