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


# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_dir = './coco'
annotation_dir = './coco/annotations'
subset = 'val2014'
im_path = os.path.join(dataset_dir, subset)
ann_path = os.path.join(annotation_dir, 'instances_{}.json'.format(subset))

imagenet_class_mappings = './imagenet_class_mappings'

imagenet_list_val = [146, 12, 137, 10, 21, 16, 14, 88, 131, 11, 985, 82, 132, 24, 136, 144, 140, 17, 15,
                     127, 143, 94, 139, 92, 96, 23, 99, 20, 346, 386, 101, 690, 349, 294, 347, 344, 369,
                     295, 379, 294, 347, 372, 342, 340]


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


class CocoDetection:
    def __init__(self, root, annFile, transform):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.categories = ['bird', 'elephant', 'cow', 'bear', 'zebra']
        self.imgIds = []
        for cats in self.categories:
            self.ids = self.coco.getCatIds(catNms=[cats])
            self.imgIds.extend(self.coco.getImgIds(catIds=self.ids))
        self.new_ids = []
        for id in self.imgIds:
            ann = (self.coco.loadAnns(self.coco.getAnnIds(id)))
            mask = self.coco.annToMask(ann[0])
            if len(ann) == 1 and mask.sum() < 10000:
                self.new_ids.append(ann[0]['image_id'])

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



torch.cuda.set_device(0)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False


COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco)

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=35, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          # sampler=RangeSampler(range(1, 16))
                                          )

print('longitud data loader:', len(data_loader))

im_label_map = imagenet_label_mappings()

textfile = open("coco_validation.txt", "w")

for i, (images, masks, paths) in enumerate(data_loader):
    images = images.cuda()
    pred = torch.nn.Softmax(dim=1)(model(images))
    pr, cl = torch.max(pred, 1)
    pred_target = cl.cpu().numpy()
    pr = pr.cpu().numpy()

    for idx, path in enumerate(paths):
        if pred_target[idx] in imagenet_list_val:
            textfile.write(path + '\n')

print('COCO Validation set created')