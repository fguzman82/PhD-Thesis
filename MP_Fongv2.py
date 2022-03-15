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
import skimage
import torchvision.transforms as transforms

# bibliotecas RISE
sys.path.insert(0, './RISE')
from evaluation import CausalMetric, auc, gkern

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# bibliotecas RISE
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
    # img_path = './dataset/0.JPEG'
    save_path = './output/'

    # gt_category = 207  # Golden retriever
    gt_category = 281  # tabby cat
    # gt_category = 258  # "Samoyed, Samoyede"
    # gt_category = 282  # tigger cat
    # gt_category = 565  # freight car
    # gt_category = 1 # goldfish, Carassius auratus
    # gt_category = 732  # camara fotografica

    try:
        shutil.rmtree(save_path)
    except OSError as e:
        print("Error: %s : %s" % (save_path, e.strerror))

    algo = 'MP'
    mask_init = 'random'
    perturb_binary = 0
    learning_rate = 0.1  # poca robustez *2 *3
    size = 224
    noise = 0.0

    max_iterations = 300
    jitter = 4
    l1_coeff = 1e-4  # poca robustez *2 *4 *0.5
    tv_beta = 3
    tv_coeff = 1e-2
    thresh = 0.5
    dataset = 'imagenet'
    save_path = './output'

    # PyTorch random seed
    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224 + jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    label_map = load_imagenet_label_map()

    # if dataset == 'imagenet':
    #     model = load_model(arch_name='resnet50')
    #
    #     # load the class label
    #     label_map = load_imagenet_label_map()
    #
    # elif dataset == 'places365':
    #     model = load_model_places365(arch_name='resnet50')
    #
    #     # load the class label
    #     label_map = load_class_label()
    #
    # else:
    #     print('Invalid datasest!!')
    #     exit(0)

    # model = torch.nn.DataParallel(model).to('cuda')
    model = models.googlenet(pretrained=True)
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
    original_img_pil = Image.open(img_path).convert('RGB')
    img_noise_np = skimage.util.random_noise(np.asarray(original_img_pil), mode='gaussian',
                                             mean=0, var=noise,
                                             )  # numpy, dtype=float64,range (0, 1)
    img_noise = Image.fromarray(np.uint8(img_noise_np * 255))

    # se normaliza la imágen y se agrega una dimensión [1,3,244,244]
    img_normal = transform(img_noise).unsqueeze(0)  # Tensor (1, 3, 224, 224)
    img_normal.requires_grad = False
    img_normal = img_normal.cuda()

    cat_orig = label_map[gt_category]

    # Path to the output folder
    save_path1 = os.path.join(save_path, '{}'.format(algo), 'pertub')
    mkdir_p(os.path.join(save_path1))
    save_path2 = os.path.join(save_path, '{}'.format(algo), 'mask')
    mkdir_p(os.path.join(save_path2))

    # Compute original output
    org_softmax = torch.nn.Softmax(dim=1)(model(img_normal))  # tensor(1,1000)
    prob_orig = org_softmax.data[0, gt_category].cpu().detach().numpy()

    o_img_path = os.path.join(save_path, 'real_{}_{:.3f}_image.jpg'
                              .format(cat_orig.split(',')[0].split(' ')[0].split('-')[0], prob_orig))

    # visualización de tensor normalizado a array y desrnomalizado
    img_transform_T = np.moveaxis(img_normal[0, :].cpu().detach().numpy().transpose(), 0, 1)  # array (224,224,3)
    img_unormalize = np.uint8(255 * unnormalize(img_transform_T))  # array (224,224,3)
    Image.fromarray(img_unormalize).save(o_img_path, 'JPEG')

    print('probabilidad original para ', cat_orig, '=', prob_orig)

    img = img_normal
    # Modified
    if mask_init == 'random':
        np.random.seed(seed=0)
        mask = np.random.rand(28, 28)
        mask = numpy_to_torch(mask)

    # imagen nulla difuminada
    orig_img_blur = transforms.GaussianBlur(kernel_size=223, sigma=10)(original_img_pil)
    # null_img_blur = orig_img_blur

    # orig_img_blur = original_img_pil.filter(ImageFilter.GaussianBlur(10))
    null_img_blur = transform(orig_img_blur).unsqueeze(0)

    null_img_blur.requires_grad = False
    null_img = null_img_blur.cuda()

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

        perturbated_input2 = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask)
        perturbated_mask = img[:, :, j1:(size + j1), j2:(size + j2)].mul(1 - upsampled_mask)


        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))

        loss = l1_coeff * torch.sum(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta) + \
               outputs[0, gt_category]
        loss.backward()

        optimizer.step()
        mask.data.clamp_(0, 1)

        # Create save_path for storing intermediate steps
        path = os.path.join(save_path1, 'intermediate_steps')
        path2 = os.path.join(save_path2, 'intermediate_steps')
        mkdir_p(path)
        mkdir_p(path2)

        # Save intermediate steps
        amax, aind = outputs.max(dim=1)
        gt_val = outputs.data[:, gt_category]
        temp_intermediate = np.uint8(
                               255 * unnormalize(
                                   np.moveaxis(perturbated_input2[0, :].cpu().detach().numpy().transpose(), 0, 1)))

        temp_intermediate2 = np.uint8(
                            255 * unnormalize(
                                np.moveaxis(perturbated_mask[0, :].cpu().detach().numpy().transpose(), 0, 1)))

        cv2.imwrite(
           os.path.abspath(os.path.join(path, 'intermediate_{:05d}_{}_{:.3f}_{}_{:.3f}.jpg'
                        .format(i, label_map[aind.item()].split(',')[0].split(' ')[0].split('-')[0],
                                amax.item(), label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0],
                                gt_val.item()))), cv2.cvtColor(temp_intermediate, cv2.COLOR_BGR2RGB))
        cv2.imwrite(
            os.path.abspath(os.path.join(path2, 'intermediate_{:05d}_{}_{:.3f}_{}_{:.3f}.jpg'
                                         .format(i, label_map[aind.item()].split(',')[0].split(' ')[0].split('-')[0],
                                                 amax.item(),
                                                 label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0],
                                                 gt_val.item()))), cv2.cvtColor(temp_intermediate2, cv2.COLOR_BGR2RGB))


    # np.save(os.path.abspath(os.path.join(save_path, "mask_{}.npy".format(algo))),
    #        1 - mask.cpu().detach().numpy()[0, 0, :])

    masked_pred = outputs[0, gt_category].cpu().detach().numpy()
    print('prediccion:', masked_pred)

    mask_np = np.squeeze(mask.cpu().detach().numpy())  # array fp32 (28, 28)
    mask_np = resize(np.moveaxis(mask_np.transpose(), 0, 1), (size, size))
    # plt.title('noise = {}'.format(noise))


    transform_eval = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_eval = transform_eval(original_img_pil).unsqueeze(0)

    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
    h = deletion.single_run(img_eval, (1. - mask_np), verbose=1)
    del_score = auc(h)
    print('deletion score: ', del_score)

    np.save('fong_{}.npy'.format(noise), (1. - mask_np))

    print('Time taken: {:.3f}'.format(time.time() - init_time))

    plt.imshow(1 - mask_np)  # 1-mask para deletion
    plt.text(175, 215, np.round(del_score, 4), color='black', fontsize=17,
             bbox=dict(facecolor='white', alpha=1, ec='white'))
    plt.text(178, 200, 'del score', color='white', fontsize=11)
    plt.text(5, 215, str(np.round(masked_pred * 100, 3)) + '%', fontsize=17, bbox=dict(boxstyle='round',
                                                                                       ec=(0., 0., 153 / 255),
                                                                                       fc=(
                                                                                           153 / 255, 221 / 255,
                                                                                           255 / 255),
                                                                                       alpha=0.8))

    plt.colorbar()
    plt.axis('off')
    plt.show()
