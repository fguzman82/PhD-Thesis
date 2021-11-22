import xml.etree.ElementTree as ET
import argparse, time, os, sys, glob, warnings, ipdb, math
from srblib import abs_path
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import glob

imagenet_val_xml_path='./val_bb'
imagenet_val_path = './val/'
base_img_dir = abs_path(imagenet_val_path)
input_dir_path = 'images_list.txt'
text_file = abs_path(input_dir_path)

img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])

def preprocess_gt_bb(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    preprocessed_img_tensor = transform(np.uint8(255 * img)).numpy()
    return preprocessed_img_tensor[0, :, :]

def get_true_bbox(img_path, base_xml_dir=abs_path(imagenet_val_xml_path)):
    # parse the xml for bounding box coordinates
    temp_img = Image.open(img_path)
    sz = temp_img.size # width x height
    im_name = img_path.split('/')[-1].split('.')[0]
    tree = ET.parse(os.path.join(abs_path(base_xml_dir), f'{im_name}.xml'))

    root = tree.getroot()
    # Get Ground Truth ImageNet masks

    # temp_area = 0
    gt_mask = np.zeros((sz[1], sz[0]))  # because we want rox x col
    for iIdx, type_tag in enumerate(root.findall('object/bndbox')):
        xmin = int(type_tag[0].text)
        ymin = int(type_tag[1].text)
        xmax = int(type_tag[2].text)
        ymax = int(type_tag[3].text)
        # if (ymax - ymin)*(xmax - xmin) > temp_area:
            # temp_area = (ymax - ymin)*(xmax - xmin)
        gt_mask[ymin:ymax, xmin:xmax] = 1

    gt = preprocess_gt_bb(gt_mask, 224)
    gt = (gt >= 0.5).astype(float) #binarize after resize

    return gt

gt = get_true_bbox(os.path.join(abs_path(imagenet_val_path), f'{img_name_list[2]}.JPEG'))

print(gt.shape)

plt.imshow(gt)
plt.show()