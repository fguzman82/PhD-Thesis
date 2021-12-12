import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model

ds = CIFAR('/path/to/cifar')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path='/path/to/model', state_dict_path='model')
model.eval()
attack_kwargs = {
   'constraint': 'inf', # L-inf PGD
   'eps': 0.05, # Epsilon constraint (L-inf norm)
   'step_size': 0.01, # Learning rate for PGD
   'iterations': 100, # Number of PGD steps
   'targeted': True # Targeted attack
   'custom_loss': None # Use default cross-entropy loss
}

_, test_loader = ds.make_loaders(workers=0, batch_size=10)
im, label = next(iter(test_loader))
target_label = (label + ch.randint_like(label, high=9)) % 10
adv_out, adv_im = model(im, target_label, make_adv, **attack_kwargs)