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
from skimage.util import view_as_windows
import skimage
from skimage.transform import resize


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
model = model.eval()
model = model.cuda()

for param in model.parameters():
    param.requires_grad = False

print('GPU 0 explicacion SP - COCO')

transform_coco = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class OcclusionAnalysis:
    def __init__(self, image, net):
        self.image = image
        self.model = net

    def explain(self, neuron, loader):
        # Compute original output
        org_softmax = torch.nn.Softmax(dim=1)(self.model(self.image))
        eval0 = org_softmax.data[0, neuron]
        batch_heatmap = torch.Tensor().cuda()

        for data in loader:
            data = data.cuda()
            softmax_out = torch.nn.Softmax(dim=1)(self.model(data * self.image))
            delta = eval0 - softmax_out.data[:, neuron]
            batch_heatmap = torch.cat((batch_heatmap, delta))

        sqrt_shape = len(loader)
        attribution = np.reshape(batch_heatmap.cpu().numpy(), (sqrt_shape, sqrt_shape))
        attribution = np.clip(attribution, 0, 1)
        attribution = resize(attribution, (size, size))
        return attribution


size = 224
patch_size = 75
stride = 3

batch_size = int((224 - patch_size) / stride) + 1

# Create all occlusion masks initially to save time
# Create mask
input_shape = (3, size, size)
total_dim = np.prod(input_shape)
index_matrix = np.arange(total_dim).reshape(input_shape)
idx_patches = view_as_windows(index_matrix, (3, patch_size, patch_size), stride).reshape(
    (-1,) + (3, patch_size, patch_size))

# Start perturbation loop
batch_size = int((size - patch_size) / stride) + 1
batch_mask = torch.zeros(((idx_patches.shape[0],) + input_shape), device='cuda')
total_dim = np.prod(input_shape)
for i, p in enumerate(idx_patches):
    mask = torch.ones(total_dim, device='cuda')
    mask[p.reshape(-1)] = 0  # occ_val
    batch_mask[i] = mask.reshape(input_shape)

trainloader = torch.utils.data.DataLoader(batch_mask.cpu(), batch_size=batch_size, shuffle=False,
                                          num_workers=0)
del mask
del batch_mask

COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco
                        )

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          # sampler=RangeSampler(range(1, 5))
                                          )

print('longitud data loader:', len(data_loader))

save_path = './output_SP_coco'

iterator = enumerate(tqdm(data_loader, total=len(data_loader), desc='batch'))

for i, (image, mask, paths) in iterator:
    image = image.cuda()
    pred = model(image)
    pr, cl = torch.max(pred, 1)
    pred_target = cl.cpu()
    pr = pr.cpu().numpy()
    heatmap_occ = OcclusionAnalysis(image, net=model)
    heatmap = heatmap_occ.explain(neuron=pred_target.item(), loader=trainloader)
    gt_masks = mask.numpy()
    mask_file = ('{}.npy'.format(paths[0].split('.jpg')[0]))
    np.save(os.path.abspath(os.path.join(save_path, mask_file)), heatmap)

