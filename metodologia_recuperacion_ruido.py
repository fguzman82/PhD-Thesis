import torch
import os
import random
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from srblib import abs_path
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import skimage
from PIL import ImageFilter, Image
from tqdm import tqdm

# buenos resultados con MP, V2, V4?
# results_path = './googlenet_v2_gen'
results_path = './output_MP_0.05'

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
    def __init__(self, data_path, transform, img_idxs=[0, 1], noise_var=0.0):
        self.data_path = data_path
        self.transform = transform
        self.noise_mean = 0
        self.noise_var = noise_var

        img_list = img_name_list[img_idxs[0]:img_idxs[1]]
        self.img_filenames = [os.path.join(data_path, '{}.JPEG'.format(i)) for i in img_list]
        self.mask_filenames = [os.path.join(results_path, '{}_mask.npy'.format(i)) for i in img_list]

    def __getitem__(self, index):
        img_orig = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        target = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        mask = np.load(self.mask_filenames[index])

        img_noise = skimage.util.random_noise(np.asarray(img_orig), mode='gaussian',
                                        mean=self.noise_mean, var=self.noise_var,
                                        )  # numpy, dtype=float64,range (0, 1)
        img_noise = Image.fromarray(np.uint8(img_noise * 255))
        img_orig = self.transform(img_orig)
        img_noise = self.transform(img_noise)
        return img_orig, img_noise, mask.reshape(1, 224, 224), target, os.path.join(self.data_path, self.img_filenames[index])
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


im_label_map = imagenet_label_mappings()

val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 25], noise_var=0.7)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=10, pin_memory=True)

torch.cuda.set_device(0)
model = models.googlenet(pretrained=True)
# model = models.resnet50(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
# model = torch.nn.DataParallel(model, device_ids=[0,1])
model.cuda()
model.eval()

for p in model.parameters():
    p.requires_grad = False


iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

# top_cnt_full = torch.empty((val_loader.batch_size, 10), dtype=torch.bool)
top_cnt_full = []

for i, (images, images_noise, masks, targets, file_names) in iterator:
    images_orig = images.cuda()
    images_noise = images_noise.cuda()
    targets_orig = targets.numpy()  # las y originales

    # predicciones originales
    preds_orig = torch.nn.Softmax(dim=1)(model(images_orig))
    probs_orig, labels_orig = torch.topk(preds_orig, 10)  # Top 10 predicciones originales

    # reenmascaramiento de img noise con explicacion
    extended_mask = masks.expand(masks.size(0), 3, 224, 224)
    noise_masked = images_noise.mul(extended_mask.cuda())
    noise_masked = noise_masked.to(torch.float32)

    # prediccion reenmascaramiento
    preds_masked = torch.nn.Softmax(dim=1)(model(images_noise))  # tensor(10,1000)
    probs_masked, labels_masked = torch.topk(preds_masked, 1000)  # predicciones recuperadas

    for i in range(val_loader.batch_size):  # se itera en el tamaño del batch
        prob_orig = probs_orig.cpu().detach()[i]    # (10, ) topk = 10
        label_orig = labels_orig.cpu().detach()[i]  # (10, ) topk = 10

        prob_masked = probs_masked.cpu().detach()[i]  # (1000,)
        label_masked = labels_masked.cpu().detach()[i]  # (1000,)

        # pred_list = [im_label_map.get(label) for label in label_orig]
        # label_orig_list = [label.item() for label in label_orig]
        # label_masked_list = [label.item() for label in label_masked[0:10]]
        pos_list = [torch.where(label_masked == label_orig_item)[0].item() for label_orig_item in label_orig]
        pos_list_tensor = torch.tensor(pos_list)
        top_cnt = [torch.where(pos_list_tensor <= i)[0].nelement() >= 1 for i in range(10)]
        top_cnt_full.append(torch.tensor(top_cnt))


        print('orig label, muestra', i, ': ', label_orig.tolist())
        print('pos orig en recuperado  ', pos_list)
        print('lista top 10 recuperados acum ', top_cnt)
        print('lista de recuperados    ', label_masked[0:10].tolist())

        print('')

# print(torch.stack(top_cnt_full))
print(torch.stack(top_cnt_full).sum(0)/val_loader.batch_size)

# inp = im[0].numpy().transpose((1, 2, 0))
# plt.imshow(inp)
# plt.axis('off')
# plt.show()
