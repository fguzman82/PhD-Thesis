import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#bibliotecas RISE
sys.path.insert(0, './RISE')
from utilsrise import *
from explanations import RISE
from evaluation import CausalMetric, auc, gkern

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = Dummy()

# Number of workers to load data
args.workers = 8
# Directory with images split into class folders.
# Since we don't use ground truth labels for saliency all images can be
# moved to one class folder.
args.datadir = './imagenet/val'
# Sets the range of images to be explained for dataloader.
args.range = range(95, 105)
# Size of imput images.
args.input_size = (224, 224)
# Size of batches for GPU.
# Use maximum number that the GPU allows.
args.gpu_batch = 125 #Máxima cantidad para una GPU

dataset = datasets.ImageFolder(args.datadir, preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range))

print('Found {: >5} images belonging to {} classes.'.format(len(dataset), len(dataset.classes)))
print('      {: >5} images will be explained.'.format(len(data_loader) * data_loader.batch_size))

# Load black box model for explanations
torch.cuda.set_device(0)
model1 = models.googlenet(pretrained=True)
model = nn.Sequential(model1, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False

# To use multiple GPUs
#model = nn.DataParallel(model)

explainer = RISE(model, args.input_size, args.gpu_batch)

# Generate masks for RISE or use the saved ones.
maskspath = 'masks.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')

def explain_all(data_loader, explainer):
    # Get all predicted labels first
    target = np.empty(len(data_loader), int)
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
        p, c = torch.max(model(img.cuda()), dim=1)
        target[i] = c[0]

    # Get saliency maps for all images in val loader
    explanations = np.empty((len(data_loader), *args.input_size))
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
        saliency_maps = explainer(img.cuda())
        explanations[i] = saliency_maps[target[i]].cpu().numpy()
    return explanations

explanations = explain_all(data_loader, explainer)

for i, (img, _) in enumerate(data_loader):
    p, c = torch.max(model(img.cuda()), dim=1)
    p, c = p[0].item(), c[0].item()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.axis('off')
    plt.title('{:.2f}% {}'.format(100 * p, get_class_name(c)))
    tensor_imshow(img[0])

    plt.subplot(122)
    plt.axis('off')
    plt.title(get_class_name(c))
    # tensor_imshow(img[0])
    sal = explanations[i]
    #plt.imshow(sal, cmap='jet', alpha=0.5)
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.imshow(sal)
    plt.show()

    deletion = CausalMetric(model1, 'del', 224, substrate_fn=torch.zeros_like)
    h = deletion.single_run(img.cpu(), sal, verbose=1)
    print('deletion score: ', auc(h))