import math
import numpy as np
import scipy as sp
import astropy.io.fits as pf
from components import CGComponent, EGComponent, DeltaComponent
from utils import get_hdu_from_hdulist, get_fits_image_info_from_hdulist,\
    degree_to_mas, _function_wrapper
import matplotlib

try:
    import pylab
except ImportError:
    pylab = None


class Jitter(object):
    def __init__(self, uvdata):
        self.baselines = uvdata.baselines
        self._lnpriors = dict()

    def set_priors(self):
        for baseline in self.baselines:
            self._lnpriors[baseline] = _function_wrapper(sp.stats.uniform.logpdf, [-5, 5], dict(),)

    def lnpr(self, p):
        n = len(self.baselines)
        lnprior = list()
        for i, par in enumerate(p):
            lnprior.append(self._lnpriors[self.baselines[i]](par))
        return sum(lnprior)


# TODO: ``Model`` subclasses can't be convolved with anything! It is
# `BasicImage`` that can be convolved.
# TODO: Keep components ordered by what?
class Model(object):
    """
    Basic class that represents general functionality of models.
    """
    def __init__(self, stokes=None):
        self._components = list()
        self._stokes = stokes

    # FIXME:
    def __str__(self):
        result = ""
        for comp in self._components:
            result += result.join([str(comp)])
            result += result.join(["\n"])
        return result

    @property
    def stokes(self):
        return self._stokes

    @stokes.setter
    def stokes(self, stokes):
        self._stokes = stokes.upper()

    # TODO: Implement adding gaussian components
    def from_hdulist(self, hdulist, ver=1):
        image_params = get_fits_image_info_from_hdulist(hdulist)
        self.stokes = image_params['stokes']
        hdu = get_hdu_from_hdulist(hdulist, extname='AIPS CC', ver=ver)
        # TODO: Need this when dealing with IDI UV_DATA extension binary table
        # dtype = build_dtype_for_bintable_data(hdu.header)
        dtype = hdu.data.dtype
        data = np.zeros(hdu.header['NAXIS2'], dtype=dtype)
        for name in data.dtype.names:
            data[name] = hdu.data[name]

        for flux, x, y in zip(data['FLUX'], data['DELTAX'] * degree_to_mas,
                              data['DELTAY'] * degree_to_mas):
            # We keep positions in mas
            component = DeltaComponent(flux, -x, -y)
            self.add_component(component)

    def from_2darray(self, image, pixsize, pixref=None, stokes='I'):
        """
        Create instance from 2D numpy array.
        :param image: 
            2D numpy array with intensity distribution.
        :param pixsize: 
            Iterable with size of single pixel in mas.
        :param pixref: (optional) 
            Iterable of coordinates of reference pixel. If ``None`` then use
            center of 2D array. (default: ``None``)
        :param stokes: (optional)
            Stokes parameters of intensity distribution. (default: ``I``)
        """
        imshape = np.shape(image)
        if pixref is None:
            pixref = (imshape[0]/2, imshape[1]/2)
        for (x, y), flux in np.ndenumerate(image):
            if flux:
                x -= pixref[0]
                y -= pixref[1]
                x *= pixsize[0]
                y *= pixsize[1]
                component = DeltaComponent(flux, x, y)
                self.add_component(component)
        self.stokes = stokes

    def from_fits(self, fname, ver=1):
        hdulist = pf.open(fname)
        self.from_hdulist(hdulist, ver)

    def from_txt(self, fname, style='difmap'):
        """
        Function that reads TXT-files with models in difmap or AIPS format.

        :param fname:
            File with components.
        :param style: (optional)
            Style of model file. ``difmap`` or ``aips``. (default: ``difmap``)
        """
        if style not in ['difmap', 'aips']:
            raise Exception
        if style == 'aips':
            raise NotImplementedError
        mdlo = open(fname)
        lines = mdlo.readlines()
        comps = list()
        for line in lines:
            if line.startswith('!'):
                continue
            line = line.strip('\n ')
            flux, radius, theta, major, axial, phi, type_, freq, spec =\
                line.split()
            x = -float(radius[:-1]) * np.sin(np.deg2rad(float(theta[:-1])))
            y = -float(radius[:-1]) * np.cos(np.deg2rad(float(theta[:-1])))
            flux = float(flux[:-1])
            if int(type_) == 0:
                comp = DeltaComponent(flux, x, y)
            elif int(type_) == 1:
                try:
                    bmaj = float(major)
                except ValueError:
                    bmaj = float(major[:-1])
                if float(axial[:-1]) == 1:
                    comp = CGComponent(flux, x, y, bmaj)
                else:
                    try:
                        e = float(axial)
                    except ValueError:
                        e = float(axial[:-1])
                    try:
                        bpa = -np.deg2rad(float(phi)) + np.pi / 2.
                    except ValueError:
                        bpa = -np.deg2rad(float(phi[:-1])) + np.pi / 2.
                    comp = EGComponent(flux, x, y, bmaj, e, bpa)
            else:
                raise NotImplementedError("Only CC, CG & EG are implemented")
            comps.append(comp)
        self.add_components(comps)

    # FIXME: Add only models with same stokes? Or implement multistokes models?
    def __add__(self, other):
        self._components.extend(other._components)

    def add_component(self, component):
        self._components.append(component)

    def add_components(self, *components):
        for component in components:
            self._components.append(component)

    def remove_component(self, component):
        self._components.remove(component)

    def remove_components(self, *components):
        for component in components:
            self._components.remove(component)

    def filter_components_by_r(self, r_min_mas, r_c=(0, 0)):
        """
        Remove all components that are closer then ``r_max_mas``.

        :param r_min_mas:
            Minimal distance of component to phase center [mas] to keep it in
            model.
        :param r_c: (optional)
            RA and DEC of the center of the circular area. If ``None`` that
            phase center of the map. (default: ``None``)
        """
        for component in self._components:
            # Components have coordinates (-RA, -DEC)
            if np.hypot(component.p[1]+r_c[0], component.p[2]+r_c[1]) < r_min_mas:
                self.remove_component(component)

    def clear_components(self):
        self._components = list()

    def clear_uv(self):
        self._uv = None

    def ft(self, uv=None):
        """
        Returns FT of model's components at specified points of uv-plane.
        """
        if uv is None:
            uv = self._uv
        ft = np.zeros(len(uv), dtype=complex)
        for component in self._components:
            ft += component.ft(uv)
        return ft

    def uvplot(self, uv, style='a&p', sym='.r', fig=None):
        """
        Plot FT of model (visibilities) vs uv-radius.
        """
        ft = self.ft(uv=uv)

        if style == 'a&p':
            a1 = np.angle(ft)
            a2 = np.real(np.sqrt(ft * np.conj(ft)))
        elif style == 're&im':
            a1 = ft.real
            a2 = ft.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        uv_radius = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        if fig is None:
            fig, axes = matplotlib.pyplot.subplots(nrows=2, ncols=1, sharex=True,
                                                   sharey=False)
        else:
            axes = fig.get_axes()

        axes[0].plot(uv_radius, a2, sym)
        if style == 'a&p':
            axes[0].set_ylim([0., 1.3 * max(a2)])
            axes[0].set_ylabel('Re, [Jy]')
            axes[1].set_ylabel('Im, [Jy]')
        axes[1].plot(uv_radius, a1, sym)
        if style == 'a&p':
            axes[1].set_ylim([-math.pi, math.pi])
            axes[0].set_ylabel('Amplitude, [Jy]')
            axes[1].set_ylabel('Phase, [rad]')
        matplotlib.pyplot.xlabel('UV-radius, wavelengths')
        fig.show()
        return fig

    @property
    def p(self):
        """
        Shortcut for parameters of model.
        """
        p = list()
        for component in self._components:
            p.extend(component.p)
        return p

    @p.setter
    def p(self, p):
        for component in self._components:
            component.p = p[:component.size]
            p = p[component.size:]

    @property
    def size(self):
        return len(self.p)

    def add_to_image(self, image, beam=None):
        """
        Add model to instances of ``Image`` subclasses.

        :param image:
            Instance of ``Image`` subclass.
        :param beam: (optional)
            Instance of ``Beam`` subclass to convolve model with beam before
            adding to image. If ``None`` then don't convolve.
        """
        for component in self._components:
            component.add_to_image(image, beam=beam)

    def substract_from_image(self, image, beam=None):
        """
        Subtract model from instances of ``Image`` subclasses.

        :param image:
            Instance of ``Image`` subclass.
        :param beam: (optional)
            Instance of ``Beam`` subclass to convolve model with beam before
            adding to image. If ``None`` then don't convolve.
        """
        for component in self._components:
            component.substract_from_image(image, beam=beam)

    # FIXME: Nonlinear models can't use AIC/BIC?
    def bic(self, uvdata, average_freq=True):
        """
        Returns BIC for current model and given instance of ``UVData`` class.

        :param uvdata:
            Instance of ``UVData`` class.
        :param average_freq: (optional)
            Boolean. Average frequency when calculating BIC? (default: ``True``)

        :return:
            Value of BIC criterion (the lower - the better)
        """
        from stats import LnLikelihood
        lnlik = LnLikelihood(uvdata, self, average_freq=average_freq)
        sample_size = 2*uvdata.n_usable_visibilities_difmap(stokes=self.stokes)
        if average_freq:
            sample_size /= uvdata.nif
        return -2. * lnlik(self.p) + self.size * sample_size

    def aic(self, uvdata, average_freq=True):
        """
        Returns AIC for current model and given instance of ``UVData`` class.

        :param uvdata:
            Instance of ``UVData`` class.
        :param average_freq: (optional)
            Boolean. Average frequency when calculating BIC? (default: ``True``)

        :return:
            Value of AIC criterion (the lower - the better)
        """
        from stats import LnLikelihood
        lnlik = LnLikelihood(uvdata, self, average_freq=average_freq)
        return -2. * lnlik(self.p) + self.size


if __name__ == "__main__":

    pass