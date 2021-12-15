# se utilizó temporalmente para pruebas de recuperación de conceptos
import argparse
import os
import random
import shutil
import time
import warnings
from srblib import abs_path
from PIL import ImageFilter, Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from skimage.transform import resize
import skimage

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# img_path = 'perro_gato.jpg'
# img_path = 'dog.jpg'
# img_path = 'example.JPEG'
# img_path = 'example_2.JPEG'
# img_path = 'goldfish.jpg'
img_path = './dataset/0.JPEG'
save_path = './output/'

# gt_category = 207  # Golden retriever
# gt_category = 281  # tabby cat
# gt_category = 258  # "Samoyed, Samoyede"
# gt_category = 282  # tigger cat
# gt_category = 565  # freight car
# gt_category = 1  # goldfish, Carassius auratus
# gt_category = 732  # camara fotografica

torch.cuda.set_device(0)  # especificar cual gpu 0 o 1
# model = models.googlenet(pretrained=True)
model = models.googlenet(pretrained=True)
model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False

imagenet_class_mappings = './imagenet_class_mappings'

def imagenet_label_mappings():
    fileName = os.path.join(imagenet_class_mappings, 'imagenet_label_mapping')
    with open(fileName, 'r') as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

im_label_map = imagenet_label_mappings()

imgs = np.load('adv_im.npy')

original_img_pil = Image.open(img_path).convert('RGB')
img_normal = transform(original_img_pil).unsqueeze(0)  # Tensor (1, 3, 224, 224)
img_normal.requires_grad = False
img_normal = img_normal.cuda()

img_batch = torch.from_numpy(imgs).cuda()
img_batch = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])(img_batch)

pred = torch.nn.Softmax(dim=1)(model(img_batch))  # tensor(1,1000)
pr, cl = torch.topk(pred, 10)

for i in range(imgs.shape[0]):
    prob = pr.cpu().detach().numpy()[i]
    pred_target = cl.cpu().detach().numpy()[i]
    # pred_list = [im_label_map.get(i) for i in pred_target]
    pred_list = [i for i in pred_target]
    print(list(zip(pred_list, np.round(prob, 4).tolist())))
    #print('prob={:.1f} cat={}'.format(prob, im_label_map.get(pred_target)))







