import numpy as np
from PIL import Image
from math import floor


def amp_filter(f, thresh):
    f[abs(f) < thresh] = 0
    return f


def extract_cs2fft(image, num_of_windows=100):
    # create a margin so we can divide the image
    image_len = len(image)
    window_len = floor(image_len / num_of_windows)
    margin = np.zeros(window_len - image_len % window_len)
    train_image = np.concatenate((image, margin))
    # since we added margin the # of windows go up by 1
    num_of_windows += 1
    step = int(len(train_image)/num_of_windows)
    # create hamming window
    hamming = np.hamming(step * 2)

    # multiplying array by j so it's complex number data type
    x_sum = 1j*np.zeros(num_of_windows)

    # 50% overlapping hamming windows
    for i in range(num_of_windows - 1):
        # apply hamming window
        smoothed = train_image[int(i * step): int((i+2) * step)] * hamming
        # apply fft to smoothed window
        x_nor = np.fft.fft(smoothed)
        # normalizing the window
        max_x = max(x_nor)
        x_nor = x_nor/max_x

        # # step only for test and debug, not required
        # train_image[int(i * step): int((i + 2) * step)] = smoothed

        # sum up fft coefficients
        x_nor = sum(x_nor)
        x_sum[i] = x_nor

    fft_sum = np.fft.fft(x_sum)
    return fft_sum
