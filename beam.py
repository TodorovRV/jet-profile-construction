import numpy as np
from scipy import signal
from utils import gaussian_beam


# * Resolution could depend on position in map or frequency. Should i associate
# instance of ``Beam`` with each pixel in ``BasicImage`` instance in such cases?
# * Beam is connected with ``CCModel`` instance naturally by deconvolution
# process? NO! Delta functions can be used in modelling without CLEAN!
# def create_clean_beam_from_fits(fname):
#     """
#     Create instance of ``CleanBeam`` from FITS-file of CLEAN image.
#     """
#     image_params = get_fits_image_info(fname)
#     return CleanBeam(image_params['bmaj'] / abs(image_params['pixsize'][0]),
#                      image_params['bmin'] / abs(image_params['pixsize'][0]),
#                      image_params['bpa'], image_params['imsize'])


# Inherite from ``Image``? Better implement more abstract class  ``BasicImage``
# (imsize)
class Beam(object):
    """
    Basic class that represents point spread function.
    """
    def __init__(self):
        self.image = None

    # FIXME: This code almost repeat ``Image.convolve``! Beam wants to inherit
    # from ``BasicImage``!
    def convolve(self, image):
        """
        Convolve ``Image`` array with image-like instance or 2D array-like.

        :param image:
            Instance of ``BasicImage`` or 2D array-like.
        """
        try:
            to_convolve = image.image
        except AttributeError:
            to_convolve = np.atleast_2d(image)
        return signal.fftconvolve(to_convolve, self.image, mode='same')


class DirtyBeam(Beam):
    """
    Class that represents point spread function.
    """
    def fit_central_part(self):
        """
        Fit central part of dirty beam and return instance of ``CleanBeam``.
        """
        raise NotImplementedError


# TODO: Refactor class - beam should't depend on any size.
class CleanBeam(Beam):
    """
    Class that represents central part of point spread function.

   """
    def __init__(self):
        super(CleanBeam, self).__init__()
        self.bmaj = None
        self.bmin = None
        self.bpa = None
        self.size = None

    @property
    def beam(self):
        return self.bmaj, self.bmin, self.bpa

    def __eq__(self, other):
        """
        Compares current instance of ``CleanBeam`` class with other instance.
        """
        return (self.bmaj == other.bmaj and self.bmin == other.bmin and
                self.bpa == other.bpa)

    def __ne__(self, other):
        """
        Compares current instance of ``CleanBeam`` class with other instance.
        """
        return (self.bmaj != other.bmaj or self.bmin != other.bmin or
                self.bpa != other.bpa)

    def _construct(self, **kwargs):
        """
        :param bmaj:
            Beam major axis [pxl].
        :param bmin:
            Beam minor axis [pxl].
        :param bpa:
            Beam positional angle [rad].
        :param size:
            Size of beam image [pxl].
        """
        self.bmaj = kwargs.pop("bmaj")
        self.bmin = kwargs.pop("bmin")
        self.bpa = kwargs.pop("bpa")
        self.size = kwargs.pop("imsize")
        self.image = gaussian_beam(self.size[0], self.bmaj, self.bmin,
                                   np.rad2deg(self.bpa) + 90., self.size[1])
