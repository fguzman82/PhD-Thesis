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
max_iterations = 301
l1_coeff = 1e-6
size = 224

torch.cuda.set_device(0)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
# model = models.resnet50(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
model.cuda()
model.eval()

print('GPU 0 explicacion ver 3 (Generation)')

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


list_of_layers = ['conv1',
                  'conv2',
                  'conv3',
                  'inception3a',
                  'inception3b',
                  'inception4a',
                  'inception4b',
                  'inception4c',
                  'inception4d',
                  'inception4e',
                  'inception5a',
                  'inception5b',
                  'fc'
                  ]

# capas para resnet50
# list_of_layers = ['relu',
#                   'layer1.0',
#                   'layer1.1',
#                   'layer1.2',
#                   'layer2.0',
#                   'layer2.1',
#                   'layer2.2',
#                   'layer2.3',
#                   'layer3.0',
#                   'layer3.1',
#                   'layer3.2',
#                   'layer3.3',
#                   'layer3.4',
#                   'layer3.5',
#                   'layer4.0',
#                   'layer4.1',
#                   'layer4.2',
#                   ]

# capas para vgg16
# list_of_layers = ['features.1',
#                   'features.3',
#                   'features.6',
#                   'features.8',
#                   'features.11',
#                   'features.13',
#                   'features.15',
#                   'features.18',
#                   'features.20',
#                   'features.22',
#                   'features.25',
#                   'features.27',
#                   'features.29'
#                   ]

# capas para alexnet
# list_of_layers = ['features.1',
#                   'features.4',
#                   'features.7',
#                   'features.9',
#                   'features.11',
#                   'classifier.2',
#                   'classifier.5'
#                   ]

activation_orig = {}


def get_activation_orig(name):
    def hook(model, input, output):
        activation_orig[name] = output

    return hook


def get_activation_mask(name):
    def hook(model, input, output):
        act_mask = output
        # print(act_mask.shape). #debug
        # print(activation_orig[name].shape) #debug
        limite_sup = (act_mask <= torch.fmax(torch.tensor(0), activation_orig[name]))
        limite_inf = (act_mask >= torch.fmin(torch.tensor(0), activation_orig[name]))
        oper = limite_sup * limite_inf
        # print('oper shape=',oper.shape). #debug
        act_mask.requires_grad_(True)
        act_mask.retain_grad()
        h = act_mask.register_hook(lambda grad: grad * oper)
        # x.register_hook(update_gradients(2))
        # activation[name]=act_mask
        # h.remove()

    return hook


def my_explanation(img_batch, max_iterations, gt_category):

    F_hook = []
    exp_hook = []

    # for module_name, module in model.named_modules():
    #     if module_name in list_of_layers:
    #         F_hook.append(module.register_forward_hook(get_activation_orig(module_name)))

    for name, layer in model.named_children():
        if name in list_of_layers:
            F_hook.append(layer.register_forward_hook(get_activation_orig(name)))

    # se calculan las activaciones para el batch de imágenes y se almacenan en la lista activation_orig
    # la funcion "feed forward" registra los hook

    org_softmax = torch.nn.Softmax(dim=1)(model(img_batch))

    # se borran los hook registrados en Feed Forward
    for fh in F_hook:
        fh.remove()

    # for module_name, module in model.named_modules():
    #     if module_name in list_of_layers:
    #         exp_hook.append(module.register_forward_hook(get_activation_mask(module_name)))

    for name, layer in model.named_children():
        if name in list_of_layers:
            exp_hook.append(layer.register_forward_hook(get_activation_mask(name)))

    for param in model.parameters():
        param.requires_grad = False

    np.random.seed(seed=0)
    mask = torch.from_numpy(np.random.uniform(0.99, 1, size=(1, 1, 224, 224)))
    mask = mask.expand(img_batch.size(0), 1, 224, 224)
    mask = mask.cuda()
    mask.requires_grad = True

    # null_img = torch.zeros(img_batch.size(0), 3, 224, 224).cuda()
    null_img_blur = transforms.GaussianBlur(kernel_size=223, sigma=10)(img_batch)
    null_img_blur.requires_grad = False
    null_img = null_img_blur.cuda()

    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in trange(max_iterations):
        extended_mask = mask.expand(img_batch.size(0), 3, 224, 224)
        perturbated_input = img_batch.mul(extended_mask) + null_img.mul(1 - extended_mask)
        perturbated_input = perturbated_input.to(torch.float32)
        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # (3,1000)

        preds = outputs[torch.arange(0, img_batch.size(0)).tolist(), gt_category.tolist()]

        loss = l1_coeff * torch.sum(torch.abs(mask), dim=(1, 2, 3)) - torch.log(preds)
        loss.backward(gradient=torch.ones_like(loss).cuda())
        # mask.grad.data = torch.nn.functional.normalize(mask.grad.data, p=float('inf'), dim=(2, 3))
        optimizer.step()
        mask.data.clamp_(0, 1)

    for eh in exp_hook:
        eh.remove()
    #
    # mask_np = (mask.cpu().detach().numpy())
    #
    # for i in range(mask_np.shape[0]):
    #     plt.imshow(mask_np[i, 0, :, :])
    #     plt.show()

    return mask

batch_size = 25 # 30
val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 100], if_noise=0, noise_var=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10,
                                         pin_memory=True)

init_time = time.time()

iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

save_path = './googlenet_v3_gen'
# save_path = './resnet50_v2'
# save_path = './vgg16_v2'
# save_path = './alexnet_v2'

for i, (images, target, file_names) in iterator:
    images.requires_grad = False
    images = images.cuda()
    mask = my_explanation(images, max_iterations, target)
    mask_np = (mask.cpu().detach().numpy())

    for idx, file_name in enumerate(file_names):
        mask_file = ('{}_mask.npy'.format(file_name.split('/')[-1].split('.JPEG')[0]))
        np.save(os.path.abspath(os.path.join(save_path, mask_file)), mask_np[idx, 0, :, :])

print('Time taken: {:.3f}'.format(time.time() - init_time))
