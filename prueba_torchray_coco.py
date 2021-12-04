import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data.sampler import Sampler
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.mask import iou, encode
from PIL import Image

dataset_dir = './coco'
annotation_dir = './coco/annotations'
subset = 'val2014'
im_path = os.path.join(dataset_dir, subset)
ann_path = os.path.join(annotation_dir, 'instances_{}.json'.format(subset))

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
        self.ids = self.coco.getCatIds(catNms=['elephant'])
        self.imgIds = self.coco.getImgIds(catIds=self.ids)
        self.new_ids = []
        for id in self.imgIds:
            ann = (self.coco.loadAnns(self.coco.getAnnIds(id)))
            mask = self.coco.annToMask(ann[0])
            if len(ann) == 1 and mask.sum() < 10000:
                self.new_ids.append(ann[0]['image_id'])
        print(self.new_ids)


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


COCO_ds = CocoDetection(root=im_path,
                        annFile=ann_path,
                        transform=transform_coco)

data_loader = torch.utils.data.DataLoader(COCO_ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          #sampler=RangeSampler(range(1, 4))
                                          )

print('longitud data loader:', len(data_loader))

for i, (image, mask, path) in enumerate(data_loader):
    tensor_imshow(image[0], mask[0, 0, :].numpy().sum())
    plt.axis('off')
    plt.imshow(mask[0, 0, :], cmap='jet', alpha=0.5)
    #print('image shape', image.shape)
    print('path ', path[0].split('.jpg')[0])
    plt.show()


# for i, (image, target) in enumerate(data_loader):
#     print(i, image.shape)
#     print(target)

# COCO_ds.coco
# image_np = np.array(image)
# plt.imshow(image_np)
# plt.show()
