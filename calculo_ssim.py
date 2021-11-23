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
import pytorch_ssim
from skimage.feature import hog
from scipy.stats import spearmanr, pearsonr
from skimage.metrics import structural_similarity as ssim

results_path1 = './output_MP'
results_path2 = './output_MP_noise_1.0'

mask_path1 = os.listdir(results_path1)
mask_path2 = os.listdir(results_path2)

mask_list1 = [i.split('_mask')[0] for i in mask_path1]
mask_list2 = [i.split('_mask')[0] for i in mask_path2]


class DataProcessing:
    def __init__(self, img_idxs=[0, 1]):

        mask_list_slice1 = mask_list1[img_idxs[0]:img_idxs[1]]
        mask_list_slice2 = mask_list2[img_idxs[0]:img_idxs[1]]
        self.mask_filenames1 = [os.path.join(results_path1, '{}_mask.npy'.format(i)) for i in mask_list_slice1]
        self.mask_filenames2 = [os.path.join(results_path2, '{}_mask.npy'.format(i)) for i in mask_list_slice2]

    def __getitem__(self, index):
        mask1 = np.load(self.mask_filenames1[index])
        mask2 = np.load(self.mask_filenames2[index])

        return mask1, mask2

    def __len__(self):
        return len(self.mask_filenames1)


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
    plt.show()


batch_size = 50
idx_start = 0
idx_end = 50
#batch_size = 10
mask_dataset = DataProcessing(img_idxs=[idx_start, idx_end])
mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)

torch.cuda.set_device(0)

# iterator = tqdm(enumerate(mask_loader), total=len(mask_loader), desc='batch')
#
# for i, (mask1, mask2) in iterator:
#     mask1 = mask1.cuda()
#     mask2 = mask2.cuda()
#     mask1 = mask1.reshape(mask1.size(0), 1, mask1.size(1), mask1.size(2))
#     mask2 = mask2.reshape(mask2.size(0), 1, mask2.size(1), mask2.size(2))
#     print('SSIM = ', pytorch_ssim.ssim(mask1, mask2).cpu().numpy())
#     # print('SSIM = ', pytorch_ssim.ssim(mask1, mask2, size_average = False).cpu().numpy())

fong0 = np.load('fong_0.0.npy')
fong1 = np.load('fong_0.05.npy')
fong2 = np.load('fong_0.1.npy')

fong_A = torch.from_numpy(np.stack((fong0, fong0)).reshape(2, 1, 224, 224))
fong_B = torch.from_numpy(np.stack((fong1, fong2)).reshape(2, 1, 224, 224))

print('SSIM (Fong) = ', pytorch_ssim.ssim(fong_A, fong_B))

fabio0 = np.load('v4_0.0.npy')
fabio1 = np.load('v4_0.05.npy')
fabio2 = np.load('v4_0.1.npy')

fabio_A = torch.from_numpy(np.stack((fabio0, fabio0)).reshape(2, 1, 224, 224))
fabio_B = torch.from_numpy(np.stack((fabio1, fabio2)).reshape(2, 1, 224, 224))

print('SSIM (Fabio) = ', pytorch_ssim.ssim(fabio_A, fabio_B))

hog0, hog_img0 = hog(fabio1, pixels_per_cell=(16, 16), visualize=True)
hog1, hog_img1 = hog(fabio2, pixels_per_cell=(16, 16), visualize=True)
out, _ = pearsonr(hog0, hog1)
print('pearson', out)

out, _ = spearmanr(fabio1, fabio2, axis=None)
print('spearman', out)

out = ssim(fabio1, fabio2, data_range=1, win_size=5)
print('ssim =', out)