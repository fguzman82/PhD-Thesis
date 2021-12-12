import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from srblib import abs_path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import skimage

#bibliotecas RISE
sys.path.insert(0, './RISE')
from utilsrise import *
from explanations import RISE

print('Explicacion RISE GPU 0')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)
imagenet_class_mappings = './imagenet_class_mappings'

torch.manual_seed(0)

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

# Size of imput images.
input_size = (224, 224)
# Size of batches for GPU.
# Use maximum number that the GPU allows.
gpu_batch = 200 # 125 #Máxima cantidad para una GPU (200 para alexnet)
# gpu_batch = 20 #Máxima cantidad para una GPU para VGG16

# Load black box model for explanations
torch.cuda.set_device(0)   # especificar cual gpu 0 o 1
# model = models.googlenet(pretrained=True)
# model = models.resnet50(pretrained=True)
# model = models.vgg16(pretrained=True)
model = models.alexnet(pretrained=True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False

explainer = RISE(model, input_size, gpu_batch)

# Generate masks for RISE or use the saved ones.
maskspath = 'masks.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')

val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 100], if_noise=0, noise_var=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10,
                                         pin_memory=True)
print('      {: >5} images will be explained.'.format(len(val_loader) * val_loader.batch_size))
# # Get all predicted labels first
# target = np.empty(len(val_loader), int)
# for i, (img, _) in enumerate(tqdm(val_loader, total=len(val_loader), desc='Predicting labels')):
#     p, c = torch.max(model(img.cuda()), dim=1)
#     target[i] = c[0]

# save_path = './resnet50_RISE'
# save_path = './vgg16_RISE'
save_path = './alexnet_RISE'

# Get saliency maps for all images in val loader
explanations = np.empty((len(val_loader), *input_size))
for i, (img, target, file_name) in enumerate(tqdm(val_loader, total=len(val_loader), desc='Explaining images')):
    saliency_maps = explainer(img.cuda())
    explanations[i] = saliency_maps[target.item()].cpu().numpy()
    mask_file = ('{}_mask.npy'.format(file_name[0].split('/')[-1].split('.JPEG')[0]))
    np.save(os.path.abspath(os.path.join(save_path, mask_file)), explanations[i])


# Rutina para graficar las explicaciones
# for i, (img, _) in enumerate(data_loader):
#     p, c = torch.max(model(img.cuda()), dim=1)
#     p, c = p[0].item(), c[0].item()
#
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121)
#     plt.axis('off')
#     plt.title('{:.2f}% {}'.format(100 * p, get_class_name(c)))
#     tensor_imshow(img[0])
#
#     plt.subplot(122)
#     plt.axis('off')
#     plt.title(get_class_name(c))
#     tensor_imshow(img[0])
#     sal = explanations[i]
#     plt.imshow(sal, cmap='jet', alpha=0.5)
#     # plt.colorbar(fraction=0.046, pad=0.04)
#
#     plt.show()