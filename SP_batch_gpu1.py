import os
import cv2
import sys
import time
import torch
import argparse
import torch.optim
import numpy as np
from srblib import abs_path
from formal_utils import *
from skimage.util import view_as_windows
from skimage.transform import resize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import skimage
from skimage.transform import resize

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

imagenet_val_xml_path = './val_bb'
imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)
imagenet_class_mappings = './imagenet_class_mappings'

torch.manual_seed(0)
print('Explicacion SP - GPU 1')

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

im_label_map = imagenet_label_mappings()


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
        return img, target, os.path.join(self.data_path, self.img_filenames[index])
        #return img, target

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


transform_val = transforms.Compose([
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

torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model.eval()
model.cuda()
label_map = load_imagenet_label_map()

for p in model.parameters():
    p.requires_grad = False

batch_size = int((224 - patch_size) / stride) + 1

# Create all occlusion masks initially to save time
# Create mask
input_shape = (3, size, size)
total_dim = np.prod(input_shape)
index_matrix = np.arange(total_dim).reshape(input_shape)
idx_patches = view_as_windows(index_matrix, (3, patch_size, patch_size), stride).reshape((-1,) + (3, patch_size, patch_size))

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

init_time = time.time()

save_path='./output_SP'

val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[501, 1001], if_noise=0, noise_var=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10,
                                         pin_memory=True)
iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

for i, (image, target, file_name) in iterator:
    image = image.cuda()
    # Occlusion class
    heatmap_occ = OcclusionAnalysis(image, net=model)
    heatmap = heatmap_occ.explain(neuron=target.item(), loader=trainloader)
    mask_file = ('{}_mask.npy'.format(file_name[0].split('/')[-1].split('.JPEG')[0]))
    np.save(os.path.abspath(os.path.join(save_path, mask_file)), heatmap)


print('Time taken: {:.3f}'.format(time.time() - init_time))