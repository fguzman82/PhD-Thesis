import argparse
import os
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
from tqdm import tqdm
import shutil
import skimage

val_dir = './val'

imagenet_val_xml_path = './val_bb'
imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)
imagenet_class_mappings = './imagenet_class_mappings'

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
    def __init__(self, data_path, transform, img_idxs=[0, 1], if_noise=0, noise_var=0):
        self.data_path = data_path
        self.transform = transform
        self.if_noise = if_noise
        self.noise_mean = 0
        self.noise_var = noise_var

        img_list = img_name_list[img_idxs[0]:img_idxs[1]]
        self.img_filenames = [os.path.join(data_path, f'{i}.JPEG') for i in img_list]
        #print('img filenames=', os.path.join(self.img_filenames[0]))

        #self.img_filenames.sort()

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        target = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))

        if self.if_noise == 1:
            img = skimage.util.random_noise(np.asarray(img), mode='gaussian',
                                            mean=self.noise_mean, var=self.noise_var,
                                            )  # numpy, dtype=float64,range (0, 1)
            img = Image.fromarray(np.uint8(img * 255))
            #print(img2.max(), img2.min())

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
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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

init_time = time.time()
val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 4], if_noise=0, noise_var=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=24, pin_memory=True)

# especificar cual gpu 0 o 1
torch.cuda.set_device(0)
model = models.googlenet(pretrained=True)
#model = torch.nn.DataParallel(model, device_ids=[0,1])
model.cuda()
model.eval()

im_label_map = imagenet_label_mappings()

iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

for i, (images, target, path) in iterator:
    images = images.cuda()
    pred = torch.nn.Softmax(dim=1)(model(images))
    target = target.numpy()

    for i, file in enumerate(path):
        print('./dataset/{}.JPEG'.format(i))
        #shutil.copyfile(file, './dataset/{}'.format(file.split('/')[-1]))
        #shutil.copyfile(file, './dataset/{}.JPEG'.format(i))

    for j, img in enumerate(images):
        target_img = target[j].item()
        pr, cl = torch.topk(pred[j], 1)
        pr = pr.cpu().detach().numpy()[0]
        cl = cl.cpu().detach().numpy()[0]
        # title = 'p={:.2f} cat={} true={}'.format(pr, im_label_map.get(cl), im_label_map.get(target_img))
        title = 'p={:.2f} cat={}'.format(pr, im_label_map.get(cl))
        tensor_imshow(img.cpu(), title=title)



print('Time taken: {:.3f}'.format(time.time() - init_time))

