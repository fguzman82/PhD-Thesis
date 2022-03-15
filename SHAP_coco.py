import torch, torchvision
from torch import nn
from torchvision import transforms, models, datasets
import shap
import json
import os
from srblib import abs_path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
from pycocotools.coco import COCO
from tqdm import tqdm, trange
from PIL import ImageFilter, Image
import skimage
from skimage.transform import resize

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_dir = './coco'
annotation_dir = './coco/annotations'
subset = 'val2014'
im_path = os.path.join(dataset_dir, subset)
ann_path = os.path.join(annotation_dir, 'instances_{}.json'.format(subset))

imagenet_class_mappings = './imagenet_class_mappings'

input_dir_path = 'coco_validation.txt'
text_file = abs_path(input_dir_path)


def imagenet_label_mappings():
    fileName = os.path.join(imagenet_class_mappings, 'imagenet_label_mapping')
    with open(fileName, 'r') as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping

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
    # plt.show()

class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(int(line.split('.jpg')[0].split('_')[-1]))


class CocoDetection:
    def __init__(self, root, annFile, transform):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.new_ids = img_name_list

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


transform_coco = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

torch.manual_seed(0)
torch.cuda.set_device(1)  # especificar cual gpu 0 o 1
print('GPU 1 explicacion SHAP - COCO')
model = models.googlenet(pretrained=True)
model.cuda()
model.eval()

im_label_map = imagenet_label_mappings()

for param in model.parameters():
    param.requires_grad = False

COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco
                        )

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          # sampler=RangeSampler(range(1, 5))
                                          )

print('longitud data loader:', len(data_loader))
im_label_map = imagenet_label_mappings()

save_path = './output_SHAP_coco'

iterator = enumerate(tqdm(data_loader, total=len(data_loader)))

for i, (image, mask, paths) in iterator:
    image = image.cuda()
    e = shap.GradientExplainer((model, model.conv2), image)
    shap_values, indexes = e.shap_values(image, ranked_outputs=1, nsamples=20)
    heatmap = np.clip(shap_values[0].sum(1), 0, 1)
    mask_file = ('{}.npy'.format(paths[0].split('.jpg')[0]))
    # np.save(os.path.abspath(os.path.join(save_path, mask_file)), resize(heatmap[0], (224, 224)))
    # tensor_imshow(image[0].cpu(), title=None)
    # plt.imshow(resize(heatmap[0], (224, 224)), cmap='jet', alpha=0.5)
    # plt.axis('off')
    # plt.show()
