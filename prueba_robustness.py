import torch as ch
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import numpy as np

ds = ImageNet('./imagenet')
model, _ = make_and_restore_model(arch='googlenet', dataset=ds, pytorch_pretrained=True)
model.eval()
model.cuda()
attack_kwargs = {
   'constraint': 'inf', # L-inf PGD
   'eps': 0.05, # Epsilon constraint (L-inf norm)
   'step_size': 0.01, # Learning rate for PGD
   'iterations': 100, # Number of PGD steps
   'targeted': True, # Targeted attack
   'do_tqdm': True,
}

_, test_loader = ds.make_loaders(workers=0, batch_size=10, only_val=True)
im, label = next(iter(test_loader))
target_label = (label + ch.randint_like(label, high=9)) % 10
adv_out, adv_im = model(im.cuda(), target_label.cuda(), make_adv=True, **attack_kwargs)

from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT

# Get predicted labels for adversarial examples
pred, _ = model(adv_im)
label_pred = ch.argmax(pred, dim=1)

np.save('adv_im.npy', adv_im.cpu().numpy())
# Visualize test set images, along with corresponding adversarial examples
show_image_row([im.cpu(), adv_im.cpu()],
         tlist=[[CLASS_DICT['ImageNet'][int(t)] for t in l] for l in [label, label_pred]],
         fontsize=18,
         filename='./adversarial_example_CIFAR.png')