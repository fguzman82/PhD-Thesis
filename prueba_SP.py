import os
import cv2
import sys
import time
import torch
import argparse
import torch.optim
import numpy as np
from formal_utils import *
from skimage.util import view_as_windows
from skimage.transform import resize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import skimage
from skimage.transform import resize

sys.path.insert(0, './RISE')
from evaluation import CausalMetric, auc, gkern

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class occlusion_analysis:
    def __init__(self, image, net, num_classes=256, img_size=227, batch_size=64,
                 org_shape=(224, 224)):
        self.image = image
        self.model = net
        self.num_classes = num_classes
        self.img_size = img_size
        self.org_shape = org_shape
        self.batch_size = batch_size

    def explain(self, neuron, loader, l_map, path='./'):

        # Compute original output
        org_softmax = torch.nn.Softmax(dim=1)(self.model(self.image))

        eval0 = org_softmax.data[0, neuron]

        batch_heatmap = torch.Tensor().cuda()

        # Create save_path for storing intermediate steps
        path = os.path.join(path, 'intermediate_steps')
        mkdir_p(path)

        for i, data in enumerate(loader):
            data = data.cuda()

            softmax_out = torch.nn.Softmax(dim=1)(self.model(data * self.image))
            delta = eval0 - softmax_out.data[:, neuron]

            # For saving intermediate steps
            amax, aind = softmax_out.max(dim=1)
            gt_val = softmax_out.data[:, neuron]

            for j in range(data.shape[0]):
                temp_img = np.uint8(255 * unnormalize(
                    np.moveaxis((data[j, :] * self.image[0, :]).cpu().detach().numpy().transpose(), 0, 1)))
                cv2.imwrite(
                    os.path.abspath(os.path.join(path, 'intermediate_{:05d}_{}_{:.3f}_{}_{:.3f}.jpg'
                                 .format(i * self.batch_size + j, l_map[aind[j].item()].split(',')[0].split(' ')[0].split('-')[0],
                                         amax[j].item(), l_map[neuron].split(',')[0].split(' ')[0].split('-')[0],
                                         gt_val[j].item()))),
                    cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

            batch_heatmap = torch.cat((batch_heatmap, delta))

        sqrt_shape = len(loader)
        attribution = np.reshape(batch_heatmap.cpu().numpy(), (sqrt_shape, sqrt_shape))

        return attribution

# img_path = 'perro_gato.jpg'
# img_path = 'dog.jpg'
# img_path = 'example.JPEG'
# img_path = 'example_2.JPEG'
img_path = 'goldfish.jpg'
# img_path = './dataset/0.JPEG'

# gt_category = 207  # Golden retriever
# gt_category = 281  # tabby cat
# gt_category = 258  # "Samoyed, Samoyede"
# gt_category = 282  # tigger cat
# gt_category = 565  # freight car
gt_category = 1 # goldfish, Carassius auratus
# gt_category = 732  # camara fotografica

algo = 'SP'
size = 224
patch_size = 35
stride = 3

dataset = 'imagenet'
save_path ='./SP_single'

model = models.googlenet(pretrained=True)
model.eval()
model.cuda()
label_map = load_imagenet_label_map()

for p in model.parameters():
    p.requires_grad = False

batch_size = int((224 - patch_size) / stride) + 1

# Create all occlusion masks initially to save time
# Create mask
input_shape = (3, size, size)
total_dim = np.prod(input_shape)
index_matrix = np.arange(total_dim).reshape(input_shape)
idx_patches = view_as_windows(index_matrix, (3, patch_size, patch_size), stride).reshape((-1,) + (3, patch_size, patch_size))

# Start perturbation loop
batch_size = int((size - patch_size) / stride) + 1
batch_mask = torch.zeros(((idx_patches.shape[0],) + input_shape), device='cuda')
total_dim = np.prod(input_shape)
for i, p in enumerate(idx_patches):
    mask = torch.ones(total_dim, device='cuda')
    mask[p.reshape(-1)] = 0  # occ_val
    batch_mask[i] = mask.reshape(input_shape)

trainloader = torch.utils.data.DataLoader(batch_mask.cpu(), batch_size=batch_size, shuffle=False,
                                          num_workers=0)
del mask
del batch_mask

init_time = time.time()

original_img = cv2.imread(img_path, 1)

shape = original_img.shape
img = np.float32(original_img) / 255

# Convert to torch variables
img = preprocess_image(img, size)

if use_cuda:
    img = img.to('cuda')

# Path to the output folder
save_path = os.path.join(save_path, '{}'.format(algo), '{}'.format(dataset))
mkdir_p(save_path)

# save original image
# Compute original output
org_softmax = torch.nn.Softmax(dim=1)(model(img))
eval0 = org_softmax.data[0, gt_category]
pill_transf = get_pil_transform()
cv2.imwrite(os.path.abspath(os.path.join(save_path, 'real_{}_{:.3f}_image.jpg'
                         .format(label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0], eval0))),
            cv2.cvtColor(np.array(pill_transf(get_image(img_path))), cv2.COLOR_BGR2RGB))


# Occlusion class
heatmap_occ = occlusion_analysis(img, net=model, num_classes=1000, img_size=size,
                                 batch_size=batch_size, org_shape=shape)
heatmap = heatmap_occ.explain(neuron=gt_category, loader=trainloader, path=save_path, l_map=label_map)
np.save(os.path.abspath(os.path.join(save_path, 'mask_{}.npy'.format(algo))), heatmap)

heatmap = np.clip(heatmap, 0, 1)
heatmap = resize(heatmap, (size, size))
print(heatmap.shape)
plt.imshow(heatmap)
plt.axis('off')
plt.show()

print('Time taken: {:.3f}'.format(time.time() - init_time))

deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
h = deletion.single_run(img.cpu(), heatmap, verbose=1)
print('deletion score: ', auc(h))