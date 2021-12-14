from __future__ import absolute_import
import warnings
warnings.simplefilter('ignore')

import time, os, sys, cv2, time, argparse
import torch
import random
import numpy as np
from formal_utils import *
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import skimage

#bibliotecas RISE
sys.path.insert(0, './RISE')
from evaluation import CausalMetric, auc, gkern

use_cuda = torch.cuda.is_available()
# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

if_pre = 0
lime_background_pixel = 0
lime_superpixel_num = 50
lime_num_samples = 500
lime_superpixel_seed = 0
lime_explainer_seed = 0
batch_size =100
true_class = gt_category
save_path = './'
dataset = 'imagenet'
algo ='LIME'

model1 = models.googlenet(pretrained=True)
model = nn.Sequential(model1, nn.Softmax(dim=1))
model.to('cuda')
model.eval()

for p in model.parameters():
    p.requires_grad = False

def get_pytorch_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf

def get_pil_transform():
    transf = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    return transf


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


if __name__ == '__main__':
    label_map = load_imagenet_label_map()
    pytorch_explainer = lime_image.LimeImageExplainer(random_state=lime_explainer_seed)
    slic_parameters = {'n_segments': lime_superpixel_num, 'compactness': 30, 'sigma': 3}
    segmenter = SegmentationAlgorithm('slic', **slic_parameters)
    pill_transf = get_pil_transform()

    #########################################################
    # Function to compute probabilities
    # Pytorch
    pytorch_preprocess_transform = get_pytorch_preprocess_transform()

    def pytorch_batch_predict(images):
        batch = torch.stack(tuple(pytorch_preprocess_transform(i) for i in images), dim=0)
        batch = batch.to('cuda')
        probs = model(batch)
        return probs.cpu().numpy()


    # Preprocess transform
    pytorch_preprocessFn = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    random.seed(0)
    init_time = time.time()

    # This image will be passed to Lime Explainer
    img = get_image(img_path)

    pytorch_img = pytorch_preprocessFn(Image.open(img_path).convert('RGB')).to('cuda').unsqueeze(0)
    outputs = model(pytorch_img)

    labels = (true_class,)

    # LIME analysis

    # save_dir
    save_path = os.path.join(save_path, '{}'.format(algo), '{}'.format(dataset))
    mkdir_p(save_path)
    # save path for intermediate steps
    save_intermediate = os.path.join(save_path, 'intermediate_steps')
    mkdir_p(save_intermediate)

    lime_img = np.array(pill_transf(img))
    t1 = time.time()
    pytorch_lime_explanation = pytorch_explainer.explain_instance(lime_img, pytorch_batch_predict,
                                                                  batch_size=batch_size,
                                                                  segmentation_fn=segmenter,
                                                                  top_labels=None, labels=labels,
                                                                  hide_color=None,
                                                                  num_samples=lime_num_samples,
                                                                  random_seed=lime_superpixel_seed,
                                                                  )
    pytorch_segments = pytorch_lime_explanation.segments
    pytorch_heatmap = np.zeros(pytorch_segments.shape)
    local_exp = pytorch_lime_explanation.local_exp
    exp = local_exp[true_class]

    for i, (seg_idx, seg_val) in enumerate(exp):
        pytorch_heatmap[pytorch_segments == seg_idx] = seg_val

    print('Time taken: {:.3f} secs'.format(time.time()-init_time))
    # print(pytorch_heatmap.shape)  #(224, 224)
    plt.imshow(pytorch_heatmap)
    plt.axis('off')
    plt.show()

    # SAVE raw numpy values
    np.save(os.path.abspath(os.path.join(save_path, "mask_{}.npy".format(algo))), pytorch_heatmap)

    # Compute original output
    org_softmax = model(pytorch_img)
    eval0 = org_softmax.data[0, true_class]
    pill_transf = get_pil_transform()
    cv2.imwrite(os.path.abspath(os.path.join(save_path, 'real_{}_{:.3f}_image.jpg'
                             .format(label_map[true_class].split(',')[0].split(' ')[0].split('-')[0], eval0))),
                cv2.cvtColor(np.array(pill_transf(get_image(img_path))), cv2.COLOR_BGR2RGB))

    deletion = CausalMetric(model1, 'del', 224, substrate_fn=torch.zeros_like)
    h = deletion.single_run(pytorch_img.cpu(), pytorch_heatmap, verbose=1)
    print('deletion score: ', auc(h))