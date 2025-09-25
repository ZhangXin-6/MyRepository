import sys

from matplotlib import pyplot as plt

sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.StereoModel import StereoModel
from core.utils.utils import InputPadder
from PIL import Image

DEVICE = 'cuda'


# def load_image(imfile):
#     img = Image.open(imfile)
#     assert len(img.split()) == 1
#     img = np.array(img).astype(np.int16)
#     img = (img - np.mean(img)) / np.std(img)
#     img = np.tile(img[..., None], (1, 1, 3))
#     img = img[..., :1]
#
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#
#     return img[None].to(DEVICE)


def load_image(imfile):
    img = Image.open(imfile)
    assert len(img.split()) == 3
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(StereoModel(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    modelName = args.restore_ckpt.split('.')[0]
    modelName = modelName.split('/')[-1]
    output_directory = Path(str(output_directory) + '/' + str(modelName))
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=64)
            image1, image2 = padder.pad(image1, image2)

            disp_up = model(image1, image2, iters=args.valid_iters, mode='test')
            disp_up = padder.unpad(disp_up).squeeze()

            file_stem = imfile1.split("\\")[-1]
            # print(file_stem)
            file_stem = file_stem.split('.')[0]
            # cmap = jet 色彩映射
            # plt.imsave(output_directory / f"{file_stem}.png", disp_up.cpu().numpy().squeeze(), cmap='jet')
            disparity = (-disp_up.cpu().numpy().squeeze())
            disparity = Image.fromarray(disparity)
            disparity.save(output_directory / f"{file_stem}.tif")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default="./models/.pth",
                        help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="datasets/US3D-Test/left/JAX_004_009_010_LEFT_RGB.tif")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="datasets/US3D-Test/right/JAX_004_009_010_RIGHT_RGB.tif")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--valid_iters', type=int, default=16,
                        help='number of flow-field updates during validation forward pass')

    # Transformer parameters

    parser.add_argument('--tf_layers', type=int, default=6, help="Transformer layer num")
    parser.add_argument('--mode', type=str, default='Stereo', choices=['SoftMax', 'Stereo'], help='mode')
    parser.add_argument('--head_num', type=int, default=6, help='head num')


    args = parser.parse_args()

    demo(args)
