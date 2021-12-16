import argparse
import os
import sys
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

# bibliotecas RISE
sys.path.insert(0, './RISE')
from evaluation import CausalMetric, auc, gkern

results_path = './output_v2'
imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
imagenet_class_mappings = './imagenet_class_mappings'
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)

mask_filenames = os.listdir(results_path)
mask_list = [i.split('_mask')[0] for i in mask_filenames]

img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])


def imagenet_label_mappings():
    fileName = os.path.join(imagenet_class_mappings, 'imagenet_label_mapping')
    with open(fileName, 'r') as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping


class DataProcessing:
    def __init__(self, data_path, transform, img_idxs=[0, 1]):
        self.data_path = data_path
        self.transform = transform

        img_list = img_name_list[img_idxs[0]:img_idxs[1]]
        #img_list = mask_list[img_idxs[0]:img_idxs[1]]

        self.img_filenames = [os.path.join(data_path, f'{i}.JPEG') for i in img_list]
        # self.img_filenames.sort()

        self.img_filenames = [os.path.join(data_path, '{}.JPEG'.format(i)) for i in img_list]
        # print('img filenames=', self.img_filenames)
        self.mask_filenames = [os.path.join(results_path, '{}_mask.npy'.format(i)) for i in img_list]
        # print('mask filenames=', self.mask_filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        target = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        mask = np.load(self.mask_filenames[index])
        img = self.transform(img)
        return img, mask, target
        # , os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        # ImageNet 2012 validation set images?
        with open(os.path.join(imagenet_class_mappings, "ground_truth_val2012")) as f:
            ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                    for x in f.readlines() if len(x.strip()) > 0}

        def get_class(f):
            ret = ground_truth_val2012.get(f, None)
            return ret

        image_class = get_class(filepath.split('/')[-1])
        return image_class


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 2
idx_start = 0 #22
idx_end = 20 #22+5
# batch_size = 10
mask_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[idx_start, idx_end])
mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size, shuffle=False, num_workers=24,
                                          pin_memory=True)

torch.cuda.set_device(0)
# model = models.googlenet(pretrained=True)
model = models.resnet50(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.cuda()
model.eval()

for p in model.parameters():
    p.requires_grad = False

im_label_map = imagenet_label_mappings()
iterator = tqdm(enumerate(mask_loader), total=len(mask_loader), desc='batch')

for i, (images, mask, target) in iterator:
    pred = torch.nn.Softmax(dim=1)(model(images.cuda()))
    for j, img in enumerate(images):
        target_img = target[j].item()
        pr, cl = torch.topk(pred[j], 1)
        pr = pr.cpu().detach().numpy()[0]
        cl = cl.cpu().detach().numpy()[0]
        title = 'p={:.1f} p={} t={}'.format(pr, im_label_map.get(cl), im_label_map.get(target_img))
        # title = 'target={}'.format(im_label_map.get(target_img))
        tensor_imshow(img, title=title)
        mask_np = mask[j].numpy()
        plt.imshow(mask_np)
        plt.axis('off')
        plt.show()
