from __future__ import absolute_import
import warnings

warnings.simplefilter('ignore')

import time, os, sys, cv2, time, argparse
import torch
import random
from srblib import abs_path
import numpy as np
from formal_utils import *
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import skimage

print('Explicacion LIME GPU 0')

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

lime_background_pixel = 0
lime_superpixel_num = 50
lime_num_samples = 500
lime_superpixel_seed = 0
lime_explainer_seed = 0
batch_size = 100

torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model.cuda()
model.eval()

for p in model.parameters():
    p.requires_grad = False

imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)
imagenet_class_mappings = './imagenet_class_mappings'

img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])

class DataProcessing:
    def __init__(self, data_path, transform, img_idxs=[0, 1], if_noise=0, noise_var=0.0):
        self.data_path = data_path
        self.transform = transform
        self.if_noise = if_noise
        self.noise_mean = 0
        self.noise_var = noise_var

        img_list = img_name_list[img_idxs[0]:img_idxs[1]]
        self.img_filenames = [os.path.join(data_path, f'{i}.JPEG') for i in img_list]
        # self.img_filenames.sort()

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        target = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))

        if self.if_noise == 1:
            img = skimage.util.random_noise(np.asarray(img), mode='gaussian',
                                            mean=self.noise_mean, var=self.noise_var,
                                            )  # numpy, dtype=float64,range (0, 1)
            img = Image.fromarray(np.uint8(img * 255))

        img = self.transform(img)
        img = np.array(img)
        return img, target, os.path.join(self.data_path, self.img_filenames[index])
        # return img, target

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
                                                                  segmentation_fn=segmenter,
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

random.seed(0)
init_time = time.time()

val_dataset = DataProcessing(base_img_dir, pill_transf, img_idxs=[501, 1001], if_noise=0, noise_var=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10,
                                         pin_memory=True)
save_path = './output_LIME'

iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

for i, (image, target, file_name) in iterator:
    mask = LIME_explanation(image, target.item())
    mask_file = ('{}_mask.npy'.format(file_name[0].split('/')[-1].split('.JPEG')[0]))
    np.save(os.path.abspath(os.path.join(save_path, mask_file)), mask)

print('Time taken: {:.3f} secs'.format(time.time() - init_time))
