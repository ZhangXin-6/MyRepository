import matplotlib.pyplot as plt
import numpy as np
import pywt
import cv2


def haar(image):
    LL, (HL, LH, HH) = pywt.dwt2(image, 'haar')
    return LL, HL, LH, HH


if __name__ == '__main__':
    image = cv2.imread("D:/zhangxin/TMF-Net/datasets/OMA/left/OMA278_001_034_LEFT_RGB.tif", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    image = (image - np.mean(image)) / np.std(image)
    LL, HL, LH, HH = haar(image)

    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(LL, cmap='gray')
    plt.title('LL')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(HL), cmap='hot')
    plt.title('HL')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(LH), cmap='hot')
    plt.title('LH')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(HH), cmap='hot')
    plt.title('HH')
    plt.axis('off')

    plt.show()

