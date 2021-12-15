import torch
import os
import random
import shutil
import time
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import numpy as np
import matplotlib.pyplot as plt
from srblib import abs_path
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import skimage
from PIL import ImageFilter, Image
from tqdm import tqdm

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
    def __init__(self, data_path, transform, img_idxs=[0, 1], if_noise=0, noise_var=0.0):
        self.data_path = data_path
        self.transform = transform
        self.if_noise = if_noise
        self.noise_mean = 0
        self.noise_var = noise_var

        img_list = img_name_list[img_idxs[0]:img_idxs[1]]
        self.img_filenames = [os.path.join(data_path, '{}.JPEG'.format(i)) for i in img_list]

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
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
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


ds = ImageNet('./imagenet')
model, _ = make_and_restore_model(arch='googlenet', dataset=ds, pytorch_pretrained=True)
model.eval()
model.cuda()

# attack_kwargs = {
#     'constraint': 'inf',  # L-inf PGD
#     'eps': 0.05,  # Epsilon constraint (L-inf norm)
#     'step_size': 0.01,  # Learning rate for PGD
#     'iterations': 100,  # Number of PGD steps
#     'targeted': True,  # Targeted attack
#     'do_tqdm': True,
# }

attack_kwargs = {
    'constraint': 'inf',  # L-inf PGD
    'eps': 1.5,  # Epsilon constraint (L-inf norm)
    'step_size': 0.01,  # Learning rate for PGD
    'iterations': 10,  # Number of PGD steps
    #'targeted': True,  # Targeted attack
    'do_tqdm': True,
}

# suave
# attack_kwargs = {
#     'constraint': '2',  # L-inf PGD
#     'eps': 20,  # Epsilon constraint (L-inf norm)
#     'step_size': 0.1,  # Learning rate for PGD
#     'iterations': 50,  # Number of PGD steps
#     'targeted': True,  # Targeted attack
#     'do_tqdm': True,
# }

im_label_map = imagenet_label_mappings()

# definiciones originales de robustness para el dataset ImageNet
# _, test_loader = ds.make_loaders(workers=0, batch_size=10, only_val=True)
# im, label = next(iter(test_loader))

# batch_size 50 máximo
val_dataset = DataProcessing(base_img_dir, transform_val, img_idxs=[0, 200], if_noise=0, noise_var=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=10, pin_memory=True)

iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc='batch')

# im, label, _ = next(iter(val_loader))
# target_label = torch.randint_like(label, high=999)
# adv_out, adv_im = model(im.cuda(), target_label.cuda(), make_adv=True, **attack_kwargs)

adv_im_full = []

for i, (images, label, path) in iterator:
    images = images.cuda()
    target_label = torch.randint_like(label, high=999).cuda()
    adv_out, adv_im = model(images, target_label, make_adv=True, **attack_kwargs)
    adv_im_full.append(adv_im.cpu().numpy())

np.save('adv_im.npy', np.array(adv_im_full).reshape(-1,3,224,224))

# visualización de la imagen adversaria [0]
# inp = np.array(adv_im_full).reshape(-1,3,224,224)[0].transpose((1, 2, 0))
# plt.imshow(inp)
# plt.axis('off')
# plt.show()



from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT

# Get predicted labels for adversarial examples
pred, _ = model(adv_im)
label_pred = torch.argmax(pred, dim=1)



# Visualize test set images, along with corresponding adversarial examples
show_image_row([images.cpu(), adv_im.cpu()],
               # tlist=[[CLASS_DICT['ImageNet'][int(t)] for t in l] for l in [label, label_pred]],
               tlist=[[int(t) for t in l] for l in [label, label_pred]],
               fontsize=18,
               filename='./adversarial_example_CIFAR.png')
