from __future__ import absolute_import
import warnings
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

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

warnings.simplefilter('ignore')

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# bibliotecas inpainter
sys.path.insert(0, './generativeimptorch')
from utils.tools import get_config, get_model_list
from model.networks import Generator

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
    def __init__(self, root, annFile, transform, transform2):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.transform2 = transform2
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
        if self.transform2 is not None:
            image2 = self.transform2(image)
        return np.array(image), mask, path, image2

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
lime_background_pixel = 0
lime_superpixel_num = 50
lime_num_samples = 1000
lime_superpixel_seed = 0
lime_explainer_seed = 0
batch_size = 100

torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False

print('GPU 0 explicacion LIME - COCO')

transform_coco = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_pytorch_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


pytorch_explainer = lime_image.LimeImageExplainer(random_state=lime_explainer_seed)
slic_parameters = {'n_segments': lime_superpixel_num, 'compactness': 30, 'sigma': 3}
segmenter = SegmentationAlgorithm('slic', **slic_parameters)
pill_transf = get_pil_transform()

#########################################################
# Function to compute probabilities
# Pytorch
pytorch_preprocess_transform = get_pytorch_preprocess_transform()


def pytorch_batch_predict(images):
    batch = torch.stack(tuple(pytorch_preprocess_transform(i) for i in images), dim=0)
    batch = batch.cuda()
    probs = model(batch)
    return probs.cpu().numpy()


def LIME_explanation(img, target):
    # This image will be passed to Lime Explainer
    labels = (target,)

    # LIME analysis
    lime_img = np.squeeze(img.numpy())

    pytorch_lime_explanation = pytorch_explainer.explain_instance(lime_img, pytorch_batch_predict,
                                                                  batch_size=batch_size,
                                                                  # segmentation_fn=segmenter,
                                                                  top_labels=None, labels=labels,
                                                                  hide_color=None,
                                                                  num_samples=lime_num_samples,
                                                                  random_seed=lime_superpixel_seed,
                                                                  )
    pytorch_segments = pytorch_lime_explanation.segments
    pytorch_heatmap = np.zeros(pytorch_segments.shape)
    local_exp = pytorch_lime_explanation.local_exp
    exp = local_exp[target]

    for i, (seg_idx, seg_val) in enumerate(exp):
        pytorch_heatmap[pytorch_segments == seg_idx] = seg_val

    return pytorch_heatmap


COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=pill_transf,
                        transform2=transform_coco)

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          # sampler=RangeSampler(range(1, 5))
                                          )

print('longitud data loader:', len(data_loader))

im_label_map = imagenet_label_mappings()
thres_vals = np.arange(0.05, 1, 0.05)
iou_table = np.zeros((len(data_loader) * data_loader.batch_size, 3))

save_path = './output_LIME_coco'

for i, (image, mask, paths, image_pred) in enumerate(data_loader):
    print(i)
    image_pred = image_pred.cuda()
    pred = model(image_pred)
    pr, cl = torch.max(pred, 1)
    pred_target = cl.cpu()
    pr = pr.cpu().numpy()
    exp_mask = LIME_explanation(image, pred_target.item())
    gt_masks = mask.numpy()

    for idx, path in enumerate(paths):
        mask_file = ('{}.npy'.format(path.split('.jpg')[0]))
        np.save(os.path.abspath(os.path.join(save_path, mask_file)), exp_mask)

