# coding:utf-8
import torch
import os
import cv2
from noise2noise_fzh import Noise2Noise
import torchvision.transforms.functional as tvF
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import base64

n2n: Noise2Noise


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    # parser.add_argument('-d', '--data', help='dataset root path', default='./test_img')

    parser.add_argument('--load-ckpt', help='load model checkpoint',
                        default='./models/n2n-epoch6-0.00323.pth')

    parser.add_argument('--pretrain-model-path', help='pretrain model path',
                        default='./models/n2n-epoch28-0.00204.pth')

    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', default=True, action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['gaussian', 'poisson', 'text', 'mc'], default='text', type=str)

    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=0.5,
                        type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)

    parser.add_argument('-c', '--crop-size', help='image crop size', default=0, type=int)

    parser.add_argument('-r', '--resize-size', help='resize size', default=640, type=int)

    parser.add_argument('--clean-targets', default=False, help='use clean targets for training', action='store_true')

    return parser.parse_args()


def resize_image(img, min_scale=320, max_scale=640):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(min_scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_scale:
        im_scale = float(max_scale) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 32 == 0 else (new_h // 32) * 32
    new_w = new_w if new_w // 32 == 0 else (new_w // 32) * 32

    # re_im = cv2.resize(img, (new_w, new_h))
    return new_h, new_w


def predict(model, img):
    model.eval()
    with torch.no_grad():
        img = img.cuda()
        # Denoise
        denoised_img = model(img)
        # print('==denoised_img.shape:', denoised_img.shape)
        denoised_t = denoised_img.cpu().squeeze(0)

        denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
        print('==denoised.size:', denoised.size)
        # denoised.save('./denoised.png')
        return denoised


def _resize(img):
    """Performs random square crop of fixed size.
    Works with list so that all items get the same cropped window (e.g. for buffers).
    """
    img = Image.fromarray(img).convert('RGB')
    img = tvF.resize(img, (640, 640))
    # w, h = img.size
    # new_h, new_w = resize_image(np.array(img))#, min(w, h), max(w, h))
    # img = tvF.resize(img, (new_w, new_h))

    source_img = tvF.to_tensor(img)
    return torch.unsqueeze(source_img, dim=0)


def preTest():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # Parse test parameters
    params = parse_args()
    print("params done")
    # Initialize model and test
    nModule = Noise2Noise(params, trainable=False, pretrain_model_path=params.pretrain_model_path)
    params.redux = False
    params.clean_targets = True
    global n2n
    n2n = nModule


def remove_tool(in_base, in_type):
    """Tests Noise2Noise."""
    # ans = []
    for i, img_code in enumerate(in_base):
        # if i < 1:
        img_data = base64.b64decode(img_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # frame = cv2.imread(img_list_path)
        img_h, img_w, _ = frame.shape
        img = _resize(frame[..., ::-1])
        denoise_img = predict(n2n.model, img)

        denoise_img = denoise_img.resize((img_w, img_h))
        # denoise_img.save(name)
        denoise_img = np.array(denoise_img)[..., ::-1]
        # Output image
        image = cv2.imencode('.' + in_type[i], denoise_img)[1]
        image_64 = str(base64.b64encode(image))[2:-1]
        return image_64


if __name__ == '__main__':
    preTest()
    # remove_tool(in_path, type)
