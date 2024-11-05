import cv2
import numpy as np
original = cv2.imread('hellohaze.webp') #hazy image
dehazed = cv2.imread('byehaze.jpg') #dehazed image
dehazed_resized = cv2.resize(dehazed, (original.shape[1], original.shape[0]))
def calculate_psnr(original, dehazed):
    mse = np.mean((original - dehazed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
psnr_value = calculate_psnr(original, dehazed_resized)
print(f'PSNR Value: {psnr_value} dB')

