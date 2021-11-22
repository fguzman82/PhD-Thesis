#version 1: DELETION WITHOUT REGU
import os
import cv2
import sys
import time
import scipy
import torch
import argparse
import numpy as np
import torch.optim
import shutil

from formal_utils import *
from skimage.transform import resize
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from torchvision import models

#sys.path.insert(0, './generativeimptorch')

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.float32(np.transpose(img.copy(), (2, 0, 1)))

    output_t = torch.from_numpy(output)
    if use_cuda:
        output_t = output_t.to('cuda')  # cuda()

    output_t.unsqueeze_(0)
    output_t.requires_grad = requires_grad
    return output_t


def numpy_to_torch2(img):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    output.unsqueeze_(0)
    return output


if __name__ == '__main__':

    img_path = 'perro_gato.jpg'
    # img_path = 'dog.jpg'
    # img_path = 'example.JPEG'
    # img_path = 'example_2.JPEG'
    save_path = './output/'

    # gt_category = 207  # Golden retriever
    gt_category = 281  # tabby cat
    # gt_category = 258  # "Samoyed, Samoyede"
    # gt_category = 282  # tigger cat
    #gt_category = 565  # freight car

    try:
        shutil.rmtree(save_path)
    except OSError as e:
        print("Error: %s : %s" % (save_path, e.strerror))

    # PyTorch random seed
    torch.manual_seed(0)

    learning_rate = 0.1  # 0.1 (preservation sparser) 0.3 (preservation dense)
    max_iterations = 301
    l1_coeff = 1e-4#5*1e-6  # 1e-4 (preservation)
    size = 224

    tv_beta = 3
    tv_coeff = 1e-2
    factorTV = 0.005  # 1(dense) o 0.5 (sparser/sharp)   #0.5 (preservation)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = models.vgg16(pretrained=True)
    # model = models.resnet50(pretrained=True)
    model = models.googlenet(pretrained=True)
    model.to(device)
    # evaluar el modelo para que sea deterministico
    model.eval()

    list_of_layers = ['conv1',
                      'conv2',
                      'conv3',
                      'inception3a',
                      'inception3b',
                      'inception4a',
                      'inception4b',
                      'inception4c',
                      'inception4d',
                      'inception4e',
                      'inception5a',
                      'inception5b',
                      'fc'
                      ]

    label_map = load_imagenet_label_map()
    # model = torch.nn.DataParallel(model).to('cuda')
    # model = model.to('cuda')

    init_time = time.time()

    # Leer la imágen del archivo
    # original_img = cv2.imread(img_path, 1)
    # img = np.float32(original_img) / 255
    original_img_pil = Image.open(img_path).convert('RGB')
    # original_np = np.array(original_img_pil)

    # normalización de acuerdo al promedio y desviación std de Imagenet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # se normaliza la imágen y se agrega una dimensión [1,3,244,244]
    img_normal = transform(original_img_pil).unsqueeze(0)  # Tensor (1, 3, 224, 224)
    img_normal.requires_grad = False
    img_normal = img_normal.to(device)

    cat_orig = label_map[gt_category]
    print('explicacion para: ', cat_orig)

    # Path to the output folder
    save_path = os.path.join(save_path, 'MP', 'imagenet')
    mkdir_p(os.path.join(save_path))

    # Compute original output
    # org_softmax = torch.nn.Softmax(dim=1)(model(preprocess_image(img, size)))
    org_softmax = torch.nn.Softmax(dim=1)(model(img_normal))  # tensor(1,1000)
    prob_orig = org_softmax.data[0, gt_category].cpu().detach().numpy()

    o_img_path = os.path.join(save_path, 'real_{}_{:.3f}_image.jpg'
                              .format(cat_orig.split(',')[0].split(' ')[0].split('-')[0], prob_orig))

    # visualización de tensor normalizado a array y desrnomalizado
    img_transform_T = np.moveaxis(img_normal[0, :].cpu().detach().numpy().transpose(), 0, 1)  # array (224,224,3)
    img_unormalize = np.uint8(255 * unnormalize(img_transform_T))  # array (224,224,3)
    Image.fromarray(img_unormalize).save(o_img_path, 'JPEG')

    print('probabilidad original para ', cat_orig, '=', prob_orig)


    for param in model.parameters():
        param.requires_grad = False

    img = img_normal  # tensor (1, 3, 224, 224)
    np.random.seed(seed=0)
    mask = np.random.uniform(0, 0.01, size=(224, 224))  # array (224, 224)  generation
    # mask = np.random.rand(224, 224)
    # mask = np.random.uniform(0.99, 1, size=(224, 224))  # array (224, 224)  preservation
    mask = numpy_to_torch(mask)  # tensor (1, 1, 224, 224)

    null_img = torch.zeros(1, 3, size, size).to(device)  # tensor (1, 3, 224, 224)

    # Definición del tipo de optimizador
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    # optimizer = torch.optim.SGD([mask], lr=learning_rate, momentum = 0.9)
    # momentum = 0.9
    # optimizer = torch.optim.SGD([mask],
    #                       lr=learning_rate,
    #                       momentum=momentum,
    #                       dampening=momentum)

    for i in range(max_iterations):
        # upsampled_mask = upsample(mask)

        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = mask.expand(1, 3, mask.size(2), mask.size(3))  # tensor (1, 3, 224, 224)
        # upsampled_mask = mask

        perturbated_input = img.mul(upsampled_mask) + null_img.mul(1 - upsampled_mask)

        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # tensor (1, 1000)

        similarity = -(org_softmax.data[0, gt_category] * torch.log(outputs[0, gt_category]))  # tensor

        #loss = l1_coeff * torch.sum(torch.abs(1 - mask)) + similarity + factorTV * tv_coeff * tv_norm(mask, tv_beta)
        loss = l1_coeff * torch.sum(torch.abs(1 - mask)) + outputs[0, gt_category]
        #loss = l1_coeff * torch.sum(torch.abs(1 - mask)) + outputs[0, gt_category] + factorTV * tv_coeff * tv_norm(mask,
        #                                                                                                       tv_beta)

        loss.backward()

        # mask_grads=np.squeeze(mask.grad.data.cpu().numpy())
        # print('max mask(grad)=', mask_grads.max())
        # print('min mask(grad)=', mask_grads.min())
        # torch.nn.utils.clip_grad_norm_(mask, 1, norm_type=float('inf'))

        # mask.grad.data = torch.nn.functional.normalize(mask.grad.data, p=float('inf'), dim=(2, 3))
        # torch.nn.utils.clip_grad_norm_(mask, 1)

        # mask_grads = np.squeeze(mask.grad.data.cpu().numpy())
        # print('max mask(grad) after clip=', mask_grads.max())
        # print('min mask(grad) after clip=', mask_grads.min())

        optimizer.step()
        #mask.data.clamp_(0, 1)  # mask tensor (1, 1, 224, 224)

        # debug visualización de la mascara
        # mask_np = np.squeeze(mask.cpu().detach().numpy())  # array fp32 (224, 224)
        # plt.imshow(1 - mask_np)  # 1-mask para deletion
        # plt.title("mask value")
        # plt.show()

        # debug visualización del gradiente de la mask
        # maskgrads_np = np.squeeze(mask.grad.data.cpu().numpy())
        # plt.imshow(maskgrads_np)
        # plt.title("mask grad")
        # plt.show()


        # control
        # upsampled_mask_control = mask.expand(1, 3, mask.size(2), mask.size(3))  # tensor (1, 3, 224, 224)
        # up_mask_np = upsampled_mask_control.cpu().detach().numpy()

        # Create save_path for storing intermediate steps
        path = os.path.join(save_path, 'intermediate_steps')
        mkdir_p(path)

        # DEBUG
        # mask_np = np.squeeze(mask.cpu().detach().numpy())  # array fp32 (224, 224)
        # print('max mask=', mask_np.max())
        # print('min mask=', mask_np.min())
        # mask_grads=np.squeeze(mask.grad.data.cpu().numpy())
        # print('max mask(grad)=', mask_grads.max())
        # print('min mask(grad)=', mask_grads.min())

        # torch.nn.utils.clip_grad_norm_(mask, 1)

        # mask_grads = np.squeeze(mask.grad.data.cpu().numpy())
        # print('max mask(grad) after clip=', mask_grads.max())
        # print('min mask(grad) after clip=', mask_grads.min())
        # plt.imshow(mask_np)
        # plt.show()
        # DEBUG

        if (i % 20) == 0:
            # Save intermediate steps
            amax, aind = outputs.max(dim=1)
            gt_val = outputs.data[:, gt_category]

            img_pert_np = perturbated_input[0, :].cpu().detach().numpy()  # array (3, 224, 224)
            img_pert_np_T = img_pert_np.transpose()  # array (224, 224, 3)
            img_pert_np_T2 = np.moveaxis(img_pert_np_T, 0, 1)  # array (224, 224, 3) se intercambian cols 0 y  1
            img_pert_unnorma = np.uint8(255 * unnormalize(img_pert_np_T2))  # array enteros (224, 224, 3)

            path_intermediate = os.path.abspath(os.path.join(path, 'intermediate_{:05d}_{}_{:.3f}_{}_{:.3f}.jpg'
                                                             .format(i,
                                                                     label_map[aind.item()].split(',')[0].split(' ')[
                                                                         0].split('-')[0],
                                                                     amax.item(),
                                                                     label_map[gt_category].split(',')[0].split(' ')[
                                                                         0].split('-')[0],
                                                                     gt_val.item())))

            Image.fromarray(img_pert_unnorma).save(path_intermediate, 'JPEG')

    # np.save(os.path.abspath(os.path.join(save_path, "mask_MP.npy")),
    #        1 - mask.cpu().detach().numpy()[0, 0, :])

    # up_mask_np = upsampled_mask.cpu().detach().numpy()
    # plt.imshow(up_mask_np[0, 0, :])
    # plt.show()
    print('prediccion:', outputs[0, gt_category].cpu().detach().numpy())

    mask_np = np.squeeze(mask.cpu().detach().numpy())  # array fp32 (224, 224)
    # mask_np_T = np.moveaxis(mask_np.transpose(), 0, 1)
    print('max mask=', mask_np.max())
    print('min mask=', mask_np.min())
    plt.imshow(1 - mask_np)  # 1-mask para deletion
    plt.show()

    print('Time taken: {:.3f}'.format(time.time() - init_time))

    original_img_pil = Image.open(img_path).convert('RGB')
    img_normal = transform(original_img_pil).unsqueeze(0)  # Tensor (1, 3, 224, 224)

    mask_tensor = numpy_to_torch2(1 - mask_np)  # tensor (1, 1, 224, 224)
    mask_expanded = mask_tensor.expand(1, 3, mask.size(2), mask.size(3))  # tensor (1, 3, 224, 224)
    null_img = torch.zeros(1, 3, size, size)
    img_masked = img_normal.mul(mask_expanded)

    # transforma de (PIL o tensor) de (1,3,224,224) a np; desnormaliza y grafica
    img_normal_np = img_masked.numpy()
    img_transform_T = np.moveaxis(img_normal_np[0, :].transpose(), 0, 1)
    img_unormalize = np.uint8(255 * unnormalize(img_transform_T))
    plt.imshow(img_unormalize)
    plt.show()

    # img_normal2 = transform(Image.fromarray(img_pert_unnorma)).unsqueeze(0)  # array -> PIL y retorna Tensor (1, 3, 224, 224)
    # img_normal2_np = img_normal2.numpy()

    # plt.imshow(img_pert_unnorma)
    # plt.show()

    org_softmax = torch.nn.Softmax(dim=1)(model(img_masked.to(device)))
    prob_orig = org_softmax.data[0, gt_category].cpu().detach().numpy()
    print('probabilidad de la mascara complemento=', prob_orig)
