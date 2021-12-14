from torch.utils.data.sampler import Sampler
from pycocotools.coco import COCO
import argparse
import os
import time
import sys
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
import skimage

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dataset_dir = './coco'
annotation_dir = './coco/annotations'
subset = 'val2014'
im_path = os.path.join(dataset_dir, subset)
ann_path = os.path.join(annotation_dir, 'instances_{}.json'.format(subset))

imagenet_class_mappings = './imagenet_class_mappings'

input_dir_path = 'coco_validation.txt'
text_file = abs_path(input_dir_path)

def imagenet_label_mappings():
    fileName = os.path.join(imagenet_class_mappings, 'imagenet_label_mapping')
    with open(fileName, 'r') as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping


transform_coco = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(int(line.split('.jpg')[0].split('_')[-1]))


class CocoDetection:
    def __init__(self, root, annFile, transform):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.new_ids = img_name_list

    def __getitem__(self, index):
        id = self.new_ids[index]
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        ann = (self.coco.loadAnns(self.coco.getAnnIds(id)))[0]
        mask = self.coco.annToMask(ann)
        if self.transform is not None:
            image = self.transform(image)
            mask = transforms.Resize((256, 256))(Image.fromarray(mask))
            mask = transforms.CenterCrop(224)(mask)
            mask = transforms.ToTensor()(mask)
            mask = torch.nn.functional.normalize(mask, p=float('inf')).int()
        return image, mask, path

    def __len__(self):
        return len(self.new_ids)


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
    # plt.show()


torch.manual_seed(0)


torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False

print('COCO Analisis')


def calculate_iou(gt_mask, exp_mask):
    # max_val = exp_mask.max()
    thres_vals = np.arange(0.05, 1, 0.05)
    # num_thres = len(thres_vals)

    out = []

    for thres in thres_vals:
        pred_mask = np.where(exp_mask > thres, 1, 0)
        mask_intersection = np.bitwise_and(gt_mask.astype(int), pred_mask.astype(int))
        mask_union = np.bitwise_or(gt_mask.astype(int), pred_mask.astype(int))
        IOU = np.sum(mask_intersection) / np.sum(mask_union)
        out.append(IOU)

    return np.array(out)


COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco)

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          sampler=RangeSampler(range(0, 86))
                                          )

print('longitud data loader:', len(data_loader))

im_label_map = imagenet_label_mappings()
thres_vals = np.arange(0.05, 1, 0.05)
iou_table = np.zeros((len(data_loader)*data_loader.batch_size, 3))

load_path = './output_v4_coco'

iterator = enumerate(tqdm(data_loader, total=len(data_loader), desc='batch'))

fig = plt.figure(figsize=(3.54,3.54), dpi=600)

for i, (image, mask, path) in iterator:
    image = image.cuda()
    pred = torch.nn.Softmax(dim=1)(model(image))
    pr, cl = torch.max(pred, 1)
    pred_target = cl.cpu().numpy()
    pr = pr.cpu().numpy()
    gt_mask = np.squeeze(mask.numpy())  # (224, 224)
    mask_file = ('{}.npy'.format(path[0].split('.jpg')[0]))
    exp_mask = np.load(os.path.abspath(os.path.join(load_path, mask_file)))  # (224, 224)
    exp_mask = exp_mask / exp_mask.max()

    iou = calculate_iou(gt_mask, exp_mask)
    iou_arg = np.argmax(iou)
    iou_table[i, 0] = int(i)
    iou_table[i, 1] = iou[iou_arg]
    iou_table[i, 2] = int(iou_arg)
    # print('path: ', path, ' iou = ', iou[iou_arg])

    # # title = 'p={:.1f} cat={}'.format(pr[idx], im_label_map.get(pred_target[idx]))
    title = 'iou = {}'.format(np.round(iou[iou_arg],3))
    tensor_imshow(image[0].cpu(), title=title)
    # tensor_imshow(image[0].cpu())
    plt.axis('off')
    exp_mask_th = np.where(exp_mask > thres_vals[iou_arg], 1, 0)
    plt.imshow(exp_mask_th, cmap='jet', alpha=0.4)
    # plt.imshow(gt_mask, cmap='jet', alpha=0.4)
    plt.show()

        # tensor_imshow(images[idx].cpu(), title='coco {}'.format(np.sum(gt_mask[idx, 0, :])))
        # plt.axis('off')
        # plt.imshow(masks[idx, 0, :], cmap='jet', alpha=0.5)
        # plt.show()
        #
        # mask_intersection = np.bitwise_and(gt_mask[idx, 0, :].astype(int), exp_mask_th.astype(int))
        # mask_union = np.bitwise_or(gt_mask[idx, 0, :].astype(int), exp_mask_th.astype(int))
        #
        # tensor_imshow(images[idx].cpu(), title='intersection {}'.format(np.sum(mask_intersection)))
        # plt.axis('off')
        # plt.imshow(mask_intersection, cmap='jet', alpha=0.5)
        # plt.show()
        #
        # tensor_imshow(images[idx].cpu(), title='union {}'.format(np.sum(mask_union)))
        # plt.axis('off')
        # plt.imshow(mask_union, cmap='jet', alpha=0.5)
        # plt.show()

print('CONSOLIDADO: ')
print(iou_table)
print(iou_table.mean(axis=0))

print('error = ', np.where(iou_table[:, 1] <= 0.4, 1, 0).sum()/len(data_loader))

# for i, (image, mask, path) in enumerate(data_loader):
#     image.requires_grad = False
#     image = image.cuda()
#     pred = torch.nn.Softmax(dim=1)(model(image))
#     pr, cl = torch.topk(pred, 1)
#     pr = pr.cpu().detach().numpy()[0][0]
#     pred_target = cl.cpu().detach().numpy()[0][0]
#     title = 'p={:.1f} cat={}'.format(pr, im_label_map.get(pred_target))
#     tensor_imshow(image[0].cpu(), title=title)
#     # plt.show()
#     mask = my_explanation(image, max_iterations, pred_target)
#     mask_np = np.squeeze(mask.cpu().detach().numpy())
#     plt.axis('off')
#     plt.imshow(mask_np, cmap='jet', alpha=0.5)
#     # print('path ', path[0].split('.jpg')[0])
#     # print('mask max ', mask.numpy().max())
#     # print('mask min ', mask.numpy().min())
#     plt.show()


# COCO_ds.coco
# image_np = np.array(image)
# plt.imshow(image_np)
# plt.show()
