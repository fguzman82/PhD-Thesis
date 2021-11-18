import os
import cv2
import sys
import time
import scipy
import torch
import argparse
import numpy as np
import torch.optim

from formal_utils import *
from skimage.transform import resize
from PIL import ImageFilter, Image
import shutil

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#bibliotecas RISE
sys.path.insert(0, './RISE')
from evaluation import CausalMetric, auc, gkern


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.to('cuda')  # cuda()

    output.unsqueeze_(0)
    output.requires_grad = requires_grad
    return output

def get_blurred_img(img, radius=10):
    img = Image.fromarray(np.uint8(img))
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred_img) / float(255)


if __name__ == '__main__':


    img_path = 'perro_gato.jpg'
    # img_path = 'dog.jpg'
    # img_path = 'example.JPEG'
    # img_path = 'example_2.JPEG'
    # img_path = 'goldfish.jpg'
    save_path = './output/'

    gt_category = 207  # Golden retriever
    # gt_category = 281  # tabby cat
    # gt_category = 258  # "Samoyed, Samoyede"
    # gt_category = 282  # tigger cat
    # gt_category = 565  # freight car
    # gt_category = 1 # goldfish, Carassius auratus

    try:
        shutil.rmtree(save_path)
    except OSError as e:
        print("Error: %s : %s" % (save_path, e.strerror))

    algo = 'MP'
    mask_init = 'random'
    perturb_binary = 0
    learning_rate = 0.1
    size = 224

    max_iterations = 300
    jitter = 4
    l1_coeff = 1e-4
    tv_beta = 3
    tv_coeff = 1e-2
    thresh = 0.5
    dataset = 'imagenet'
    save_path = './output'

    # PyTorch random seed
    torch.manual_seed(0)



    if dataset == 'imagenet':
        model = load_model(arch_name='resnet50')

        # load the class label
        label_map = load_imagenet_label_map()

    elif dataset == 'places365':
        model = load_model_places365(arch_name='resnet50')

        # load the class label
        label_map = load_class_label()

    else:
        print('Invalid datasest!!')
        exit(0)

    #model = torch.nn.DataParallel(model).to('cuda')
    model.to('cuda')
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if algo == 'MPG':
        # Tensorflow CA-inpainter from FIDO
        sys.path.insert(0, './generative_inpainting')
        from CAInpainter import CAInpainter

        inpaint_model = CAInpainter(1, checkpoint_dir=weight_file)

    if use_cuda:
        upsample = torch.nn.UpsamplingNearest2d(size=(size, size)).to('cuda')

    else:
        upsample = torch.nn.UpsamplingNearest2d(size=(size, size))

    init_time = time.time()

    # Read image
    original_img = cv2.imread(img_path, 1)

    shape = original_img.shape
    img_orig = np.float32(original_img) / 255

    # Path to the output folder
    save_path = os.path.join(save_path, '{}'.format(algo), '{}'.format(dataset))
    mkdir_p(os.path.join(save_path))

    # Compute original output
    org_softmax = torch.nn.Softmax(dim=1)(model(preprocess_image(img_orig, size)))
    eval0 = org_softmax.data[0, gt_category]
    pill_transf = get_pil_transform()
    o_img_path = os.path.join(save_path, 'real_{}_{:.3f}_image.jpg'
                              .format(label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0], eval0))
    cv2.imwrite(os.path.abspath(o_img_path), cv2.cvtColor(np.array(pill_transf(get_image(img_path))), cv2.COLOR_BGR2RGB))

    # Convert to torch variables
    img = preprocess_image(img_orig, size + jitter)

    if use_cuda:
        img = img.to('cuda')

    # Modified
    if mask_init == 'random':
        np.random.seed(seed=0)
        mask = np.random.rand(28, 28)
        mask = numpy_to_torch(mask)


    if algo == 'MP':
        null_img = preprocess_image(get_blurred_img(np.float32(original_img), radius=10), size + jitter)

    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in range(max_iterations):
        if jitter != 0:
            j1 = np.random.randint(jitter)
            j2 = np.random.randint(jitter)
        else:
            j1 = 0
            j2 = 0

        upsampled_mask = upsample(mask)

        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        perturbated_input = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask) + \
                                    null_img[:, :, j1:(size + j1), j2:(size + j2)].mul(
                                        1 - upsampled_mask)

        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))

        loss = l1_coeff * torch.sum(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta) + \
               outputs[0, gt_category]
        loss.backward()

        optimizer.step()
        mask.data.clamp_(0, 1)

        # Create save_path for storing intermediate steps
        #path = os.path.join(save_path, 'intermediate_steps')
        #mkdir_p(path)

        # Save intermediate steps
        #amax, aind = outputs.max(dim=1)
        #gt_val = outputs.data[:, gt_category]
        #temp_intermediate = np.uint8(
        #                        255 * unnormalize(
        #                            np.moveaxis(perturbated_input[0, :].cpu().detach().numpy().transpose(), 0, 1)))
        #cv2.imwrite(
        #    os.path.abspath(os.path.join(path, 'intermediate_{:05d}_{}_{:.3f}_{}_{:.3f}.jpg'
        #                 .format(i, label_map[aind.item()].split(',')[0].split(' ')[0].split('-')[0],
        #                         amax.item(), label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0],
        #                         gt_val.item()))), cv2.cvtColor(temp_intermediate, cv2.COLOR_BGR2RGB))

    #np.save(os.path.abspath(os.path.join(save_path, "mask_{}.npy".format(algo))),
    #        1 - mask.cpu().detach().numpy()[0, 0, :])


    mask_np = np.squeeze(mask.cpu().detach().numpy())  # array fp32 (28, 28)
    mask_np = resize(np.moveaxis(mask_np.transpose(), 0, 1),(size, size))
    plt.imshow(1 - mask_np)  # 1-mask para deletion
    plt.show()

    img_eval = preprocess_image(img_orig, 224).cpu()

    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
    h = deletion.single_run(img_eval, (1. - mask_np), verbose=1)
    print('deletion score: ', auc(h))

    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    # Function that blurs input image
    blur = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen // 2)

    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    h = insertion.single_run(img_eval, (1. - mask_np), verbose=1)
    print('insertion score: ', auc(h))


    print('Time taken: {:.3f}'.format(time.time() - init_time))