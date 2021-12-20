import argparse
import os
import random
import shutil
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
from matplotlib import cm
from matplotlib.colors import ListedColormap
from tqdm import tqdm, trange
import skimage
from skimage.transform import resize

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# bibliotecas inpainter
sys.path.insert(0, './generativeimptorch')
from utils.tools import get_config, get_model_list
from model.networks import Generator

imagenet_val_xml_path = './val_bb'
imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list2.txt'
text_file = abs_path(input_dir_path)
imagenet_class_mappings = './imagenet_class_mappings'

torch.manual_seed(0)
learning_rate = 0.1
max_iterations = 301
l1_coeff = 1e-3/10  # 1e-3/3  # 1e-3/10  # 1e-6 # 2*1e-7
size = 224

tv_beta = 3
tv_coeff = 1e-2
factorTV = 1  # 0.1   # 1(dense) o 0.5 (sparser/sharp)   #0.5 (preservation)


def tv_norm(input, tv_beta):
    img = input[:, 0, :]
    row_grad = torch.abs((img[:, :-1, :] - img[:, 1:, :])).pow(tv_beta).sum(dim=(1, 2))
    col_grad = torch.abs((img[:, :, :-1] - img[:, :, 1:])).pow(tv_beta).sum(dim=(1, 2))
    return row_grad + col_grad


torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
model = models.googlenet(pretrained=True)
# model = models.resnet50(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False

print('GPU 0 Metod. Recuperacion ver 2')


def imagenet_label_mappings():
    fileName = os.path.join(imagenet_class_mappings, 'imagenet_label_mapping')
    with open(fileName, 'r') as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping


im_label_map = imagenet_label_mappings()

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


# Creating colormap
uP = cm.get_cmap('Blues_r', 129)
dowN = cm.get_cmap('Blues_r', 128)
newcolors = np.vstack((
    dowN(np.linspace(0, 1, 128)),
    uP(np.linspace(0, 1, 129))
))
cMap = ListedColormap(newcolors, name='RedsBlues')
cMap.colors[257 // 2, :] = [1, 1, 1, 1]


def my_explanation(img_batch, max_iterations, gt_category):
    np.random.seed(seed=0)
    mask = torch.from_numpy(np.random.uniform(0.99, 1, size=(1, 1, 224, 224)))
    mask = mask.expand(img_batch.size(0), 1, 224, 224)
    mask = mask.cuda()
    mask.requires_grad = True

    null_img = torch.zeros(img_batch.size(0), 3, 224, 224).cuda()
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in trange(max_iterations):
        extended_mask = mask.expand(img_batch.size(0), 3, 224, 224)
        perturbated_input = img_batch.mul(extended_mask) + null_img.mul(1 - extended_mask)
        perturbated_input = perturbated_input.to(torch.float32)
        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # (3,1000)

        preds = outputs[torch.arange(0, img_batch.size(0)).tolist(), gt_category.tolist()]

        loss = l1_coeff * torch.sum(torch.abs(mask), dim=(1, 2, 3)) - (preds) + factorTV * tv_coeff * tv_norm(mask,
                                                                                                              tv_beta)
        loss.backward(gradient=torch.ones_like(loss).cuda())
        # mask.grad.data = torch.nn.functional.normalize(mask.grad.data, p=float('inf'), dim=(2, 3))
        optimizer.step()
        mask.data.clamp_(0, 1)

    mask_np = (mask.cpu().detach().numpy())

    for i in range(mask_np.shape[0]):
        # plt.imshow(mask_np[i, 0, :, :], interpolation='none', cmap=cMap)
        # plt.axis('off')
        # plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        inp = img_batch[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        ax.imshow(inp)
        # title = file_names[i].split('/')[-1].split('.JPEG')[0]
        # ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(1, 3, 2)
        mask_rz = mask_np[i, 0, :, :]
        ax.imshow(mask_rz, cmap=cMap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(1, 3, 3)
        img_masked = np.multiply(inp, np.repeat(np.expand_dims(mask_rz, axis=2), 3, axis=2))
        ax.imshow(img_masked)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.show()

    return mask


init_time = time.time()

img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])


class DataProcessing:
    def __init__(self, data_path, transform, img_idxs=[0, 1], noise_var=0.0):
        self.data_path = data_path
        self.transform = transform
        self.noise_mean = 0
        self.noise_var = noise_var

        img_list = img_name_list[img_idxs[0]:img_idxs[1]]
        self.img_filenames = [os.path.join(data_path, f'{i}.JPEG') for i in img_list]
        # self.img_filenames.sort()

    def __getitem__(self, index):
        img_orig = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        target = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))

        img_noise = skimage.util.random_noise(np.asarray(img_orig), mode='gaussian',
                                              mean=self.noise_mean, var=self.noise_var,
                                              )  # numpy, dtype=float64,range (0, 1)
        img_noise = Image.fromarray(np.uint8(img_noise * 255))

        img_orig = self.transform(img_orig)
        img_noise = self.transform(img_noise)
        return img_orig, img_noise, target, os.path.join(self.data_path, self.img_filenames[index])
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


transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 7], noise_var=1.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=7, shuffle=False, num_workers=10,
                                         pin_memory=True)

iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

recov_top_cnt_full = []
pertb_top_cnt_full = []
pertb_radio = []
recov_radio = []

for i, (imgs_orig, imgs_noise, targets, file_names) in iterator:
    imgs_orig.requires_grad = False
    imgs_orig = imgs_orig.cuda()
    imgs_noise = imgs_noise.cuda()

    # predicciones originales para comparar
    preds_orig = torch.nn.Softmax(dim=1)(model(imgs_orig))
    probs_orig, labels_orig = torch.topk(preds_orig, 10)  # Top 10 predicciones originales

    preds_noise = torch.nn.Softmax(dim=1)(model(imgs_noise))  # tensor(n_batch,1000)
    probs_noise, labels_noise = torch.topk(preds_noise, 1000)  # Top 1000 predicciones ruido

    # obtención de explicaciones para imágenes con ruido
    _, orig_labels = torch.max(preds_orig, 1)  # labels originales para generar las explicaciones, top 1
    mask = my_explanation(imgs_noise, max_iterations, orig_labels)

    # reenmascaramiento de img ruidosa con explicación
    noise_masked = imgs_noise.mul(mask)
    noise_masked = noise_masked.to(torch.float32)

    # prediccion reenmascaramiento
    preds_masked = torch.nn.Softmax(dim=1)(model(noise_masked))  # tensor(n_batch,1000)
    probs_masked, labels_masked = torch.topk(preds_masked, 1000)  # predicciones recuperadas y ordenadas (n_batch, 1000)

    for i in range(val_loader.batch_size):  # se itera en el tamaño del batch
        prob_orig = probs_orig.cpu().detach()[i]  # (10, ) topk = 10
        label_orig = labels_orig.cpu().detach()[i]  # (10, ) topk = 10

        prob_masked = probs_masked.cpu().detach()[i]  # (1000,)
        label_masked = labels_masked.cpu().detach()[i]  # (1000,)
        label_noise = labels_noise.cpu().detach()[i]

        prob_noise = probs_noise.cpu().detach()[i]

        # se buscan donde estan las etiquetas originales dentro de las perturbadas
        pertub_pos_list = [torch.where(label_noise == label_orig_item)[0].item() for label_orig_item in label_orig]
        pertub_pos_list_tensor = torch.tensor(pertub_pos_list)
        pertb_radio.append(pertub_pos_list_tensor)
        pertb_top_cnt = [torch.where(pertub_pos_list_tensor <= i)[0].nelement() >= 1 for i in range(10)]
        pertb_top_cnt_full.append(torch.tensor(pertb_top_cnt))

        # se buscan donde estan las etiquetas originales dentro de las recuperadas
        recov_pos_list = [torch.where(label_masked == label_orig_item)[0].item() for label_orig_item in label_orig]
        recov_pos_list_tensor = torch.tensor(recov_pos_list)
        recov_radio.append(recov_pos_list_tensor)
        # el indice de pos_list_tensor determina el rango de predicciones a incluir en el top 10
        # por ejemplo pos_list_tensor[0] analiza el top 10 de la primera prediccion original en el grupo de recuperados
        # pos_list_tensor[0:4] analiza el top 10 del grupo de las 3 primeras predicciones originales en el grupo recup
        recov_top_cnt = [torch.where(recov_pos_list_tensor <= i)[0].nelement() >= 1 for i in range(10)]
        recov_top_cnt_full.append(torch.tensor(recov_top_cnt))

        print('muestra ', file_names[i].split('/')[-1].split('.JPEG')[0])
        print('top 10 muestra original: ', label_orig.tolist())
        print('top 10 muestra original (decod): ', [im_label_map.get(label) for label in label_orig.tolist()])
        print('top 10 prob muestra original (%): ', [round(num*100, 2) for num in prob_orig.tolist()])
        print('lista top 10 recuperados acum ', recov_top_cnt)
        print('top 10 perturbados    ', label_noise[0:10].tolist())
        print('top 10 perturbados (decod) ', [im_label_map.get(label) for label in label_noise[0:10].tolist()])
        print('top 10 prob muestra pertb (%)', [round(num*100, 2) for num in prob_noise[0:10].tolist()])
        print('pos orig en perturbado  ', pertub_pos_list)
        print('top 10 recuperados    ', label_masked[0:10].tolist())
        print('top 10 recuperados (decod) ', [im_label_map.get(label) for label in label_masked[0:10].tolist()])
        print('top 10 prob muestra recup (%)', [round(num*100, 2) for num in prob_masked[0:10].tolist()])
        print('pos orig en recuperado  ', recov_pos_list)
        print('')

recov_table = torch.stack(recov_top_cnt_full)
recov_stats = recov_table.sum(0) / (val_loader.batch_size * len(val_loader))

pertb_table = torch.stack(pertb_top_cnt_full)
pertb_stats = pertb_table.sum(0) / (val_loader.batch_size * len(val_loader))

print('estado despues de perturbar')
print(pertb_stats)
print('')

print('estado despues de recuperar')
print(recov_stats)
print('')

pr = torch.stack(pertb_radio).float()
print('radio perturbacion')
print(pr)
print('promedio')
print(pr.mean(0))
print('')

rr = torch.stack(recov_radio).float()
print('radio recuperacion')
print(rr)
print('promedio')
print(rr.mean(0))
print('')
print('Time taken: {:.3f}'.format(time.time() - init_time))
