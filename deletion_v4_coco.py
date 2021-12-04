
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
    #plt.show()


torch.manual_seed(0)
learning_rate = 0.1 * 0.8  # orig (0.3) 0.1 (preservation sparser) 0.3 (preservation dense)
max_iterations = 228  # 130 *2
l1_coeff = 0.01e-5 * 2  # *2 *4 *0.5 (robusto)
size = 224

tv_beta = 3
tv_coeff = 1e-2
factorTV = 1 * 0.5 * 0.005  # 1(dense) o 0.5 (sparser/sharp)   #0.5 (preservation)


def inpainter(img, mask):
    config = get_config('./generativeimptorch/configs/config.yaml')
    checkpoint_path = os.path.join('./generativeimptorch/checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    cuda = config['cuda']
    device_ids = config['gpu_ids']

    with torch.no_grad():  # enter no grad context
        # Test a single masked image with a given mask
        x = img
        # denormaliza imagenet y se normaliza a inpainter [-1,1] mean=0.5, std=0.5
        x = transforms.Normalize(mean=[0.015 / 0.229, 0.044 / 0.224, 0.094 / 0.225],
                                 std=[0.5 / 0.229, 0.5 / 0.224, 0.5 / 0.225])(x)
        x = x * (mask)
        # Define the trainer
        netG = Generator(config['netG'], cuda, device_ids)
        # Resume weight
        last_model_name = get_model_list(checkpoint_path, "gen", iteration=0)
        netG.load_state_dict(torch.load(last_model_name))

        #netG = torch.nn.parallel.DataParallel(netG, device_ids=[0, 1])
        netG.cuda()
        # Inference
        x1, x2, offset_flow = netG(x, (1.-mask))

    return x2

def tv_norm(input, tv_beta):
    img = input[:, 0, :]
    row_grad = torch.abs((img[:, :-1, :] - img[:, 1:, :])).pow(tv_beta).sum(dim=(1, 2))
    col_grad = torch.abs((img[:, :, :-1] - img[:, :, 1:])).pow(tv_beta).sum(dim=(1, 2))
    return row_grad + col_grad

torch.cuda.set_device(0)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False

print('GPU 0 explicacion ver 4 COCO')


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

    for name, layer in model.named_children():
        if name in list_of_layers:
            F_hook.append(layer.register_forward_hook(get_activation_orig(name)))

    # se calculan las activaciones para el batch de imágenes y se almacenan en la lista activation_orig
    # la funcion "feed forward" registra los hook

    org_softmax = torch.nn.Softmax(dim=1)(model(img_batch))

    # se borran los hook registrados en Feed Forward
    for fh in F_hook:
        fh.remove()

    for name, layer in model.named_children():
        if name in list_of_layers:
            exp_hook.append(layer.register_forward_hook(get_activation_mask(name)))

    for param in model.parameters():
        param.requires_grad = False

    np.random.seed(seed=0)
    mask = torch.from_numpy(np.float32(np.random.uniform(0, 0.01, size=(1, 1, 224, 224))))
    mask = mask.expand(img_batch.size(0), 1, 224, 224)
    mask = mask.cuda()
    mask.requires_grad = True

    #null_img = torch.zeros(img_batch.size(0), 3, 224, 224).cuda()
    #null_img_blur = transforms.GaussianBlur(kernel_size=223, sigma=10)(img_batch)
    #null_img_blur.requires_grad = False
    #null_img = null_img_blur.cuda()


    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in trange(max_iterations):
        extended_mask = mask.expand(img_batch.size(0), 3, 224, 224)

        img_inpainted = inpainter(img_batch, mask)
        img_inpainted = transforms.Normalize(mean=-1, std=2)(img_inpainted)
        img_inpainted = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(img_inpainted)

        perturbated_input = img_batch.mul(extended_mask) + img_inpainted.mul(1 - extended_mask)
        #perturbated_input = perturbated_input.to(torch.float32)
        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # (3,1000)

        preds = outputs[torch.arange(0, img_batch.size(0)).tolist(), gt_category.tolist()]

        loss = l1_coeff * torch.sum(torch.abs(1 - mask), dim=(1, 2, 3)) + preds + \
               factorTV * tv_coeff * tv_norm(mask, tv_beta)

        loss.backward(gradient=torch.ones_like(loss).cuda())
        optimizer.step()
        mask.data.clamp_(0, 1)

    for eh in exp_hook:
        eh.remove()

    # Para visualizar las máscaras
    # mask_np = (mask.cpu().detach().numpy())
    #
    # for i in range(mask_np.shape[0]):
    #     plt.imshow(1 - mask_np[i, 0, :, :])
    #     plt.show()

    return mask


COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco)

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=35, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          #sampler=RangeSampler(range(1, 16))
                                          )

print('longitud data loader:', len(data_loader))

im_label_map = imagenet_label_mappings()

for i, (images, masks, paths) in enumerate(data_loader):
    images = images.cuda()
    pred = torch.nn.Softmax(dim=1)(model(images))
    pr, cl = torch.max(pred, 1)
    pred_target = cl.cpu().numpy()
    pr = pr.cpu().numpy()
    # exp_mask = my_explanation(images, max_iterations, pred_target)

    for idx, path in enumerate(paths):
        print(idx, path)
        title = 'p={:.1f} cat={}'.format(pr[idx], im_label_map.get(pred_target[idx]))
        tensor_imshow(images[idx].cpu(), title=title)
        plt.axis('off')
        # plt.imshow(1-exp_mask[idx, 0, :].cpu().detach().numpy(), cmap='jet', alpha=0.5)
        #plt.show()

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


# for i, (image, target) in enumerate(data_loader):
#     print(i, image.shape)
#     print(target)

# COCO_ds.coco
# image_np = np.array(image)
# plt.imshow(image_np)
# plt.show()
