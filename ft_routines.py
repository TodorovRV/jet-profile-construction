import math
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift


def fft_convolve2d(x, y):
    """ 2D convolution, using FFT."""
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m, n = fr.shape
    cc = np.real(ifft2(fr * fr2))
    # This used  to return convolution result.
    # cc = np.roll(cc, -m/2 + 1, axis=0)
    # cc = np.roll(cc, -n/2 + 1, axis=1)
    return cc


def fft_convolve1d(x, y):
    """ 1D convolution, using FFT."""
    fr = fft(x)
    fr2 = fft(np.flipud(y))
    cc = np.real(ifft(fr * fr2))
    return fftshift(cc)


def image_ft(image, x, y, u, v):
    """
    Function that returns FT of image in user specified `uv`-points.

    :param image:
        Numpy 2D array of image.
    :param x:
        Iterable of x-coordinates.
    :param y:
        Iterable of y-coordinates.
    :param u:
        Iterable of u-spatial frequencies
    :param v:
        Iterable of v-spatial frequencies
    :return:
        Numpy array of complex visibilities.
    """
    assert len(u) == len(v)
    yy, xx = np.where(image != 0)
    mask = image == 0
    image = np.ma.array(image, mask=mask)
    xx = x[xx]
    yy = y[yy]
    visibilities = list()
    image_compressed = image.compressed()
    for i, u0, v0 in zip(range(len(u)), u, v):
        # if not i % 10:
        #     print("Doing {}th uv-point".format(i))
        visibility = (image_compressed * np.exp(-2.0 * math.pi * 1j *
                                                (u0 * xx + v0 * yy))).sum()
        visibilities.append(visibility)
    return np.asarray(visibilities)