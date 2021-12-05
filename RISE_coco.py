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

# bibliotecas RISE
sys.path.insert(0, './RISE')
from utilsrise import *
from explanations import RISE

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
# Size of imput images.
input_size = (224, 224)
# Size of batches for GPU.
# Use maximum number that the GPU allows.
gpu_batch = 125 #Máxima cantidad para una GPU

torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for param in model.parameters():
    param.requires_grad = False

print('GPU 0 explicacion RISE - COCO')

transform_coco = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

explainer = RISE(model, input_size, gpu_batch)

# Generate masks for RISE or use the saved ones.
maskspath = 'masks.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')


COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco
                        )

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          # sampler=RangeSampler(range(1, 5))
                                          )

print('longitud data loader:', len(data_loader))

im_label_map = imagenet_label_mappings()
thres_vals = np.arange(0.05, 1, 0.05)
iou_table = np.zeros((len(data_loader) * data_loader.batch_size, 3))

save_path = './output_RISE_coco'

iterator = enumerate(tqdm(data_loader, total=len(data_loader)))
explanations = np.empty((len(data_loader), *input_size))

for i, (image, mask, paths) in iterator:
    image = image.cuda()
    pred = model(image)
    pr, cl = torch.max(pred, 1)
    pred_target = cl.cpu()
    pr = pr.cpu().numpy()
    saliency_maps = explainer(image)
    explanations[i] = saliency_maps[pred_target.item()].cpu().numpy()
    gt_masks = mask.numpy()
    mask_file = ('{}.npy'.format(paths[0].split('.jpg')[0]))
    np.save(os.path.abspath(os.path.join(save_path, mask_file)), explanations[i])

