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
from tqdm import tqdm, trange
from skimage.transform import resize
import skimage

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

val_dir = './val'

imagenet_val_xml_path = './val_bb'
imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)
imagenet_class_mappings = './imagenet_class_mappings'

torch.manual_seed(0)
learning_rate = 0.1
size = 224
max_iterations = 300
jitter = 4
l1_coeff = 1e-4
tv_beta = 3
tv_coeff = 1e-2
thresh = 0.5

torch.cuda.set_device(0)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
#model = torch.nn.DataParallel(model, device_ids=[0,1])
model.cuda()
model.eval()

print('GPU 0 explicacion MP')

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
    def __init__(self, data_path, transform, img_idxs=[0, 1], if_noise=0):
        self.data_path = data_path
        self.transform = transform
        self.if_noise = if_noise
        self.noise_mean = 0
        self.noise_var = 1.0

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
    transforms.CenterCrop(224+jitter),
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
    plt.show()


upsample = torch.nn.UpsamplingNearest2d(size=(size, size)).to('cuda')

def tv_norm(input, tv_beta):
    img = input[:, 0, :]
    row_grad = torch.abs((img[:, :-1, :] - img[:, 1:, :])).pow(tv_beta).sum(dim=(1,2))
    col_grad = torch.abs((img[:, :, :-1] - img[:, :, 1:])).pow(tv_beta).sum(dim=(1,2))
    return row_grad + col_grad

for param in model.parameters():
    param.requires_grad = False

def my_explanation(img_batch, max_iterations, gt_category):

    np.random.seed(seed=0)
    mask = torch.from_numpy(np.random.uniform(0, 0.01, size=(1, 1, 28, 28)))
    mask = mask.expand(img_batch.size(0), 1, 28, 28)
    mask = mask.cuda()
    mask.requires_grad = True

    #null_img = torch.zeros(img_batch.size(0), 3, 224, 224).cuda()
    null_img_blur = transforms.GaussianBlur(kernel_size=223, sigma=10)(img_batch)
    null_img_blur.requires_grad = False
    null_img = null_img_blur.cuda()

    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in trange(max_iterations):

        if jitter != 0:
            j1 = np.random.randint(jitter)
            j2 = np.random.randint(jitter)
        else:
            j1 = 0
            j2 = 0

        upsampled_mask = upsample(mask)

        extended_mask = upsampled_mask.expand(img_batch.size(0), 3, 224, 224)
        perturbated_input = img_batch[:, :, j1:(size + j1), j2:(size + j2)].mul(extended_mask) + \
                            null_img[:, :, j1:(size + j1), j2:(size + j2)].mul(1 - extended_mask)
        perturbated_input = perturbated_input.to(torch.float32)
        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # (3,1000)

        preds = outputs[torch.arange(0, img_batch.size(0)).tolist(), gt_category.tolist()]

        loss = l1_coeff * torch.sum(torch.abs(1 - mask), dim=(1, 2, 3)) + preds + tv_coeff * tv_norm(mask, tv_beta)
        loss.backward(gradient=torch.ones_like(loss).cuda())
        # mask.grad.data = torch.nn.functional.normalize(mask.grad.data, p=float('inf'), dim=(2, 3))
        optimizer.step()
        mask.data.clamp_(0, 1)

    #mask_np = (mask.cpu().detach().numpy())

    #for i in range(mask_np.shape[0]):
    #    plt.imshow(1 - mask_np[i, 0, :, :])
    #    plt.show()

    return mask

batch_size = 50
#batch_size = 10
val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 50], if_noise=1)
#val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 10])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10,
                                         pin_memory=True)

init_time = time.time()

iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

save_path='./output_MP_noise_1.0'

for i, (images, target, file_names) in iterator:
    images.requires_grad = False
    images = images.cuda()
    mask = my_explanation(images, max_iterations, target)
    mask_np = (mask.cpu().detach().numpy())

    for idx, file_name in enumerate(file_names):
        mask_file = ('{}_mask.npy'.format(file_name.split('/')[-1].split('.JPEG')[0]))
        mask_np_idx = resize(np.moveaxis(mask_np[idx, 0, :, :].transpose(), 0, 1), (size, size))
        np.save(os.path.abspath(os.path.join(save_path, mask_file)), 1 - mask_np_idx)

print('Time taken: {:.3f}'.format(time.time() - init_time))