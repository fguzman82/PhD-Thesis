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
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# bibliotecas RISE
sys.path.insert(0, './RISE')
from evaluation import CausalMetric, auc, gkern

# results_path_SP = './resnet50_SHAP'
# results_path_LIME = './resnet50_LIME'
# results_path_RISE = './resnet50_RISE'
# results_path_MP = './resnet50_MP'
# results_path_v4 = './resnet50_v2'

results_path_SP = './googlenet_SHAP'
results_path_LIME = './output_LIME'
results_path_RISE = './output_RISE'
results_path_MP = './output_MP'
results_path_v4 = './output_v2'

imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
imagenet_class_mappings = './imagenet_class_mappings'
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30. / 255, 136. / 255, 229. / 255, l))
for l in np.linspace(0, 1, 100):
    colors.append((9. / 255, 97. / 255, 3. / 255, l))
# for l in np.linspace(0, 1, 100):
#     colors.append((30./255, 136./255, 229./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

img_subset = [0, 2, 3, 5, 7, 14, 20, 24, 25, 33, 35, 32, 39, 47, 48, 49, 56, 59, 60, 77, 80, 111, 120, 121, 125,
              126, 127, 128, 132, 155, 165, 166]

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

        img_list = [img_name_list[i] for i in img_subset[img_idxs[0]:img_idxs[1]]]
        # img_list = img_name_list[img_idxs[0]:img_idxs[1]]

        self.img_filenames = [os.path.join(data_path, '{}.JPEG'.format(i)) for i in img_list]
        self.mask_filenames_SP = [os.path.join(results_path_SP, '{}_mask.npy'.format(i)) for i in img_list]
        self.mask_filenames_LIME = [os.path.join(results_path_LIME, '{}_mask.npy'.format(i)) for i in img_list]
        self.mask_filenames_RISE = [os.path.join(results_path_RISE, '{}_mask.npy'.format(i)) for i in img_list]
        self.mask_filenames_MP = [os.path.join(results_path_MP, '{}_mask.npy'.format(i)) for i in img_list]
        self.mask_filenames_v4 = [os.path.join(results_path_v4, '{}_mask.npy'.format(i)) for i in img_list]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        target = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        mask_SP = np.load(self.mask_filenames_SP[index])
        mask_LIME = np.load(self.mask_filenames_LIME[index])
        mask_RISE = np.load(self.mask_filenames_RISE[index])
        mask_MP = np.load(self.mask_filenames_MP[index])
        mask_v4 = np.load(self.mask_filenames_v4[index])

        img = self.transform(img)
        return img, mask_SP, mask_LIME, mask_RISE, mask_MP, mask_v4, target, os.path.join(self.data_path,
                                                                                          self.img_filenames[index])

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
    plt.show()


transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 1
idx_start = 20  # 22
idx_end = 31  # 22+5
# batch_size = 10
mask_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[idx_start, idx_end])
mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size, shuffle=False, num_workers=24,
                                          pin_memory=True)

torch.cuda.set_device(0)
model = models.googlenet(pretrained=True)
# model = models.resnet50(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.cuda()
model.eval()

for p in model.parameters():
    p.requires_grad = False

im_label_map = imagenet_label_mappings()
iterator = tqdm(enumerate(mask_loader), total=len(mask_loader), desc='batch')

df = pd.read_pickle('auc_scores.pkl')
cols_list = ['vgg16_SHAP', 'vgg16_LIME', 'vgg16_v4', 'vgg16_MP', 'vgg16_RISE']

titles = ['SHAP', 'LIME', 'MIC (ours)', 'MP', 'RISE']


def plot_masks(nrows, ncols, mask_arr, orig_img, file_name, prob_orig, prob_arr):
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3))
    filename = file_name.split('/')[-1].split('.JPEG')[0]
    df2 = df[df.file == filename]
    # print(df2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=0.03, hspace=0.12)
    axes = np.reshape(axes, (nrows, ncols))
    index = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row][col]
            if row == 0 and col == 0:  # imagen original
                inp = orig_img.numpy().transpose((1, 2, 0))
                # Mean and std for ImageNet
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                im = ax.imshow(inp, interpolation='none')
                inp = (0.2989 * inp[:, :, 0] + 0.5870 * inp[:, :, 1] + 0.1140 * inp[:, :, 2])  # rgb to gray
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_title('Orig Image')
                ax.text(10, 187, 'orig prob', color='black', fontsize=7,
                        bbox=dict(boxstyle='round', fc=(255 / 255, 255 / 255, 204 / 255),
                                  ec=(255 / 255, 255 / 255, 204 / 255), alpha=0.7))
                ax.text(10, 210, str(np.round(prob_orig * 100, 1)) + '%', color='black',
                        bbox=dict(boxstyle='round', fc=(255 / 255, 255 / 255, 204 / 255),
                                  ec=(255 / 255, 255 / 255, 204 / 255), alpha=0.7))
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.imshow(inp, cmap=plt.get_cmap('gray'), alpha=0.2)
                abs_vals = np.stack([np.abs(mask_arr[index][i]) for i in range(len(mask_arr[index]))], 0).flatten()
                max_val = np.nanpercentile(abs_vals, 99.9)
                im = ax.imshow(mask_arr[index], cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)
                # im = ax.imshow(mask_arr[index], interpolation='none')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_title(titles[index])
                ax.text(165, 210, np.round(df2[cols_list[index]].item(), 3), color='black',
                        bbox=dict(facecolor='white', alpha=0.8))
                ax.text(165, 190, 'del score', color='black', fontsize=7)
                ax.text(10, 210, str(np.round(prob_arr[index]*100, 1))+'%', bbox=dict(boxstyle='round',
                                                                                      ec=(0., 0., 153/255),
                                                                                      fc=(153/255, 221/255, 255/255),
                                                                                      alpha=0.5))
                ax.set_xticks([])
                ax.set_yticks([])
                index = index + 1
    plt.show()


def pred_mask(img, mask, target):
    img_masked = img.mul(1. - mask/mask.max()).cuda()
    img_masked = img_masked.to(torch.float32)
    prob = torch.nn.Softmax(dim=1)(model(img_masked)).data[0, target].cpu().numpy()
    return prob


for i, (images, mask_SP, mask_LIME, mask_RISE, mask_MP, mask_v4, target, file_name) in iterator:
    pred = torch.nn.Softmax(dim=1)(model(images.cuda()))
    prob_orig = pred.data[0, target].cpu().numpy()
    prob_SP = pred_mask(images, mask_SP, target)
    prob_LIME = pred_mask(images, mask_LIME, target)
    prob_RISE = pred_mask(images, mask_RISE, target)
    prob_MP = pred_mask(images, mask_MP, target)
    prob_v4 = pred_mask(images, mask_v4, target)
    mask_arr = []
    prob_arr = []
    for j, img in enumerate(images):
        mask_arr.append(mask_SP[j].numpy())
        mask_arr.append(mask_LIME[j].numpy())
        mask_arr.append(mask_v4[j].numpy())
        mask_arr.append(mask_MP[j].numpy())
        mask_arr.append(mask_RISE[j].numpy())
        prob_arr.append(prob_SP[0])
        prob_arr.append(prob_LIME[0])
        prob_arr.append(prob_v4[0])
        prob_arr.append(prob_MP[0])
        prob_arr.append(prob_RISE[0])
        target_img = target[j].item()
        pr, cl = torch.topk(pred[j], 1)
        pr = pr.cpu().detach().numpy()[0]
        cl = cl.cpu().detach().numpy()[0]
        # title = 'p={:.1f} p={} t={}'.format(pr, im_label_map.get(cl), im_label_map.get(target_img))
        # tensor_imshow(img, title=title)
        # mask_np = mask_LIME[j].numpy()
        # plt.imshow(mask_np)
        # plt.show()
        plot_masks(2, 3, mask_arr, img, file_name[j], prob_orig[0], prob_arr)
