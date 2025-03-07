import astropy.io.fits as pf
from model import Model
from image import Image, CleanImage


def create_model_from_fits_file(fname, ver=1):
    """
    Function that creates instance of ``Model`` from FITS-file with CLEAN model.

    :param fname:
        FITS-file with CLEAN model.
    :param ver: (optional)
        Version of AIPS CC table to use. (default: ``1``)
    :return:
        Instance of ``Model`` class.
    """
    model = Model()
    model.from_fits(fname, ver)
    return model


def create_model_from_hdulist(hdulist, ver=1):
    """
    Function that creates instance of ``Model`` from instance of
    ``PyFits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.
    :param ver: (optional)
        Version of AIPS CC table to use. (default: ``1``)
    :return:
        Instance of ``Model`` class.
    """
    model = Model()
    model.from_hdulist(hdulist, ver)
    return model


def create_clean_image_from_fits_file(fname, ver=1):
    """
    Create instance of ``CleanImage`` from FITS-file with CLEAN image.

    :param fname:
        FITS-file with CLEAN model.
    :param ver: (optional)
        Version of AIPS CC table to use. (default: ``1``)
    :return:
        Instance of ``CleanImage``.
    """
    hdulist = pf.open(fname)
    return create_clean_image_from_hdulist(hdulist, ver)


def create_clean_image_from_hdulist(hdulist, ver=1):
    """
    Function that creates instance of ``CleanImage`` from instance of
    ``Pyfits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.
    :param ver: (optional)
        Version of AIPS CC table to use. (default: ``1``)
    :return:
        Instance of ``CleanImage``.
    """
    image = CleanImage()
    image.from_hdulist(hdulist, ver)
    return image


def create_image_from_fits_file(fname):
    """
    Create instance of ``Image`` from FITS-file with image in primary HDU.

    :param fname:
        FITS-file with image in primary HDU.
    :return:
        Instance of ``Image``.
    """
    hdulist = pf.open(fname)
    return create_image_from_hdulist(hdulist)


def create_image_from_hdulist(hdulist):
    """
    Create instance of ``Image`` from instance of ``PyFits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.
    :return:
        Instance of ``Image``.
    """
    image = Image()
    image.from_hdulist(hdulist)
    return image