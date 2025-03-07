import math
import numpy as np
from utils import _function_wrapper, mas_to_rad, vcomplex, gaussian
from ft_routines import image_ft
from utils import transform_image

try:
    import pylab
except ImportError:
    pylab = None


# TODO: Make parnames and size - class attributes
class Component(object):
    """
    Basic class that implements single component of model.
    """
    _parnames = ['flux', 'x', 'y']
    size = len(_parnames)

    def __init__(self):
        self._p = None
        self._fixed = np.array([False, False, False])
        self._lnprior = dict()

    def __len__(self):
        return len(self._parnames)

    def __str__(self):
        result = "{} : ".format(self.__class__.__name__)
        for parname, parvalue in zip(self.parnames, self.p):
            result += (" {} = {:.3f},".format(parname, parvalue))
        return str(result[:-1])

    def __eq__(self, other):
        """
        Compares current instance of ``Component`` class with other instance.
        """
        return np.all(self.p == other.p) and np.all(self._p == other._p)

    def __ne__(self, other):
        """
        Compares current instance of ``Component`` class with other instance.
        """
        return np.any(self.p != other.p) or np.any(self._p != other._p)

    @property
    def parnames(self):
        return np.array(self._parnames)[~self._fixed]

    def add_prior(self, **lnprior):
        """
        Add prior for some parameters.
        :param lnprior:
            Kwargs with keys - name of the parameter and values - (callable,
            args, kwargs,) where args & kwargs - additional arguments to
            callable. Each callable is called callable(p, *args, **kwargs).
        Example:
        {'flux': (scipy.stats.uniform, [0., 10.], dict(),),
         'bmaj': (scipy.stats.uniform, [0, 5.], dict(),),
         'e': (scipy.stats.beta.logpdf, [alpha, beta], dict(),)}
        First key will result in calling: scipy.stats.norm.logpdf(x, mu, s) as
        prior for ``flux`` parameter.
        """
        for key, value in lnprior.items():
            if key in self._parnames:
                func, args, kwargs = value
                self._lnprior.update({key: _function_wrapper(func, args,
                                                             kwargs)})
            else:
                raise Exception("Uknown parameter name: " + str(key))

    # TODO: properties must return only free parameters!
    @property
    def p(self):
        """
        Shortcut for parameters of model.
        """
        return self._p[np.logical_not(self._fixed)]

    @p.setter
    def p(self, p):
        # print "Setting comp's parameters with ", p
        # print "Before: ", self._p
        self._p[np.logical_not(self._fixed)] = p[:]
        # print "After: ", self._p

    def is_within_radec(self, ra_range, dec_range):
        # Coordinates in mas
        ra_mas = -self.p[1]
        dec_mas = -self.p[2]
        return ((ra_range[0] < ra_mas < ra_range[1]) and
                (dec_range[0] < dec_mas < dec_range[1]))

    def is_within(self, blc, trc, image):
        """
        If component center contained within rectangular box defined by ``blc``
        and ``trc``
        :param blc:
            Pair of Bottom Left Corner coordinates [pixels].
        :param trc:
            Pair of Top Right Corner coordinates [pixels].
        :param image:
            Instance of ``Image`` class that deifnes mapping between physical
            coordinates and pixels using `_convert_coordinate` method.
        :return:
            Boolean.
        """
        # Coordinates in mas
        ra_mas = -self.p[1]
        dec_mas = -self.p[2]
        dec_range, ra_range = image._convert_array_bbox(blc, trc)
        return ((ra_range[0] < ra_mas < ra_range[1]) and
                (dec_range[0] < dec_mas < dec_range[1]))

    def ft(self, uv):
        """
        Method that returns Fourier Transform of component in given points of
        uv-plane.
        :param uv:
            2D-numpy array of uv-coordinates with shape (#data, 2,)
        :return:
            Numpy array (length = length(uv)) with complex visibility values.
        """
        raise NotImplementedError("Method must me implemented in subclasses!")

    def add_to_image(self, image, beam=None):
        """
        Add component to image.

        :param image:
            Instance of ``Image`` class
        :param beam: (optional)
            Instance of ``Beam`` subclass to convolve component with beam before
            adding to image. If ``None`` then don't convolve.
        """
        raise NotImplementedError("Method must me implemented in subclasses!")

    def uvplot(self, uv, style='a&p', sym='.k'):
        """
        Plot FT of component vs uv-radius.
        """
        ft = self.ft(uv)

        if style == 'a&p':
            a1 = np.angle(ft)
            a2 = np.real(np.sqrt(ft * np.conj(ft)))
        elif style == 're&im':
            a1 = ft.real
            a2 = ft.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        uv_radius = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        pylab.subplot(2, 1, 1)
        pylab.plot(uv_radius, a2, sym)
        pylab.subplot(2, 1, 2)
        pylab.plot(uv_radius, a1, sym)
        if style == 'a&p':
            pylab.ylim([-math.pi, math.pi])
        pylab.show()

    @property
    def lnpr(self):
        if not self._lnprior.keys():
            print("No priors specified")
            return 0
        lnprior = list()
        for key, value in self._lnprior.items():
            # Index of ``key`` in ``_parnames`` the same as in ``_p``
            par = self._p[self._parnames.index(key)]
            lnprior.append(value(par))
        return sum(lnprior)


# TODO: Use mod(pi) for BPA?
class EGComponent(Component):
    """
    Class that implements elliptical gaussian component.
    """
    _parnames = ['flux', 'x', 'y', 'bmaj', 'e', 'bpa']
    size = len(_parnames)

    def __init__(self, flux, x, y, bmaj, e, bpa, fixed=None):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component phase center [mas].
        :param y:
            Y-coordinate of component phase center [mas].
        :param bmaj:
            Std of component size [mas].
        :param e:
            Minor-to-major axis ratio.
        :param bpa:
            Positional angle of major axis. Angle counted from x-axis of image
            plane counter clockwise [rad].
        :param fixed: (optional)
            Iterable of parameter's names that are fixed. If ``None`` then all
            parameters are not fixed. (default: ``None``)
        """
        super(EGComponent, self).__init__()
        self._fixed = np.concatenate((self._fixed,
                                      np.array([False, False, False]),))
        self._p = np.array([flux, x, y, bmaj, e, bpa])
        self.size = 6
        if fixed is not None:
            for par in fixed:
                if par not in self._parnames:
                    raise Exception('Uknown parameter ' + str(par) + ' !')
                self._fixed[self._parnames.index(par)] = True
            self.size = 6 - np.count_nonzero(self._fixed)

    def to_delta(self, fixed=None):
        """
        Create a delta function component from current elliptical.

        :param fixed: (optional)
            Iterable of parameter's names that are fixed. If ``None`` use
            parameters of elliptical components.
        :return:
            Instance of ``DeltaComponent``
        """
        if fixed is None:
            fixed = self._fixed[:DeltaComponent.size]
        return DeltaComponent(self.p[0], self.p[1], self.p[2],
                              fixed=np.array(DeltaComponent._parnames)[fixed])

    def to_circular(self, fixed=None):
        """
        Create a circular gaussian component from current elliptical.

        :param fixed: (optional)
            Iterable of parameter's names that are fixed. If ``None`` use
            parameters of elliptical components.
        :return:
            Instance of ``CGcomponent``.
        """
        if fixed is None:
            fixed = self._fixed[:CGComponent.size]
        return CGComponent(self.p[0], self.p[1], self.p[2], self.p[3],
                           fixed=np.array(CGComponent._parnames)[fixed])

    def ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.

        :note:

            The value of the Fourier transform of gaussian function (Wiki):

            g(x, y) = A*exp[-(a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2)]  (1)

            where:

                a = cos(\theta)**2/(2*std_x**2)+sin(\theta)**2/(2*std_y**2)
                b = sin(2*\theta)/(2*std_x**2)-sin(2*\theta)/(2*std_y**2)
                (corresponds to rotation counter clockwise)
                c = sin(\theta)**2/(2*std_x**2)+cos(\theta)**2/(2*std_y**2)

            For x0=0, y0=0 in point u,v of uv-plane is (Briggs Thesis):

            2*pi*A*(4*a*c-b**2)**(-1/2)*exp[(4*pi**2/(4*a*c-b**2))*
                    (-c*u**2+b*u*v-a*v**2)] (2)

            As we parametrize the flux as full flux of gaussian (that is flux at
            zero (u,v)-spacing), then change coefficient in front of exponent to
            A.

            Shift of (x0, y0) in image plane corresponds to phase shift in
            uv-plane:

            ft(x0,y0) = ft(x0=0,y0=0)*exp(-2*pi*(u*x0+v*y0))
        """
        try:
            flux, x0, y0, bmaj, e, bpa = self._p
            if e == 0:
                e = 10**(-54)
        # If we call method inside ``CGComponent``
        except ValueError:
            # Jy, mas, mas, mas
            flux, x0, y0, bmaj = self._p
            e = 1.
            bpa = 0.

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad
        bmaj *= mas_to_rad
        bpa += 0.5 * np.pi

        u = uv[:, 0]
        v = uv[:, 1]
        c = (np.pi*bmaj)**2/(4. * np.log(2.))
        b = e**2 * (u*np.cos(bpa)-v*np.sin(bpa))**2 + (u*np.sin(bpa)+v*np.cos(bpa))**2
        ft = flux*np.exp(-c*b)
        ft = vcomplex(ft)
        # If x0=!0 or y0=!0 then shift phase accordingly
        if x0 or y0:
            ft *= np.exp(-2. * math.pi * 1j * (u * x0 + v * y0))
        return ft

    def _ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.

        :note:

            The value of the Fourier transform of gaussian function (Wiki):

            g(x, y) = A*exp[-(a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2)]  (1)

            where:

                a = cos(\theta)**2/(2*std_x**2)+sin(\theta)**2/(2*std_y**2)
                b = sin(2*\theta)/(2*std_x**2)-sin(2*\theta)/(2*std_y**2)
                (corresponds to rotation counter clockwise)
                c = sin(\theta)**2/(2*std_x**2)+cos(\theta)**2/(2*std_y**2)

            For x0=0, y0=0 in point u,v of uv-plane is (Briggs Thesis):

            2*pi*A*(4*a*c-b**2)**(-1/2)*exp[(4*pi**2/(4*a*c-b**2))*
                    (-c*u**2+b*u*v-a*v**2)] (2)

            As we parametrize the flux as full flux of gaussian (that is flux at
            zero (u,v)-spacing), then change coefficient in front of exponent to
            A.

            Shift of (x0, y0) in image plane corresponds to phase shift in
            uv-plane:

            ft(x0,y0) = ft(x0=0,y0=0)*exp(-2*pi*(u*x0+v*y0))
        """
        try:
            flux, x0, y0, bmaj, e, bpa = self._p
            if e == 0:
                e = 10**(-54)
        # If we call method inside ``CGComponent``
        except ValueError:
            # Jy, mas, mas, mas
            flux, x0, y0, bmaj = self._p
            e = 1.
            bpa = 0.

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad
        bmaj *= mas_to_rad

        u = uv[:, 0]
        v = uv[:, 1]
        # Construct parameter of gaussian function (1)
        std_x = bmaj / (2. * np.sqrt(2. * np.log(2)))
        std_y = e * bmaj / (2. * np.sqrt(2. * np.log(2)))
        # std_x = bmaj / (2. * np.sqrt(2. * np.log(2)))
        # std_y = e * bmaj / (2. * np.sqrt(2. * np.log(2)))
        a = math.cos(bpa) ** 2. / (2. * std_x ** 2.) + \
            math.sin(bpa) ** 2. / (2. * std_y ** 2.)
        b = math.sin(2. * bpa) / (2. * std_x ** 2.) - \
            math.sin(2. * bpa) / (2. * std_y ** 2.)
        c = math.sin(bpa) ** 2. / (2. * std_x ** 2.) + \
            math.cos(bpa) ** 2. / (2. * std_y ** 2.)
        a = np.double(a)
        b = np.double(a)
        c = np.double(a)
        flux = np.double(flux)
        # Calculate the value of FT in point (u,v) for x0=0,y0=0 case using (2)
        k = (4. * a * c - b ** 2.)
        ft = flux * np.exp((4. * math.pi ** 2. / k) * (-c * u ** 2. +
                                                       b * u * v - a * v ** 2.))
        # ft = flux * np.exp((4. * math.pi ** 2.) * (-(c/k) * u ** 2. +
        #                                            (b/k) * u * v - (a/k) * v ** 2.))
        ft = vcomplex(ft)
        # If x0=!0 or y0=!0 then shift phase accordingly
        if x0 or y0:
            ft *= np.exp(-2. * math.pi * 1j * (u * x0 + v * y0))
        return ft

    def add_to_image(self, image, beam=None):
        """
        Add component to given instance of ``Image`` class.
        """
        # Cellsize [rad]
        dx, dy = image.dx, image.dy
        # Center of image [pix]
        x_c, y_c = image.x_c, image.y_c

        # Parameters of component
        try:
            # Jy, mas, mas, mas,  , rad
            flux, x0, y0, bmaj, e, bpa = self._p
        # If we call method inside ``CGComponent``
        except ValueError:
            flux, x0, y0, bmaj = self._p
            e = 1.
            bpa = 0.

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad
        bmaj *= mas_to_rad

        # TODO: Is it [Jy/beam]??
        # Amplitude of gaussian component [Jy/beam]
        # amp = flux / (2. * math.pi * (bmaj / mas_to_rad) ** 2. * e)
        # amp = flux / (2. * math.pi * (bmaj / abs(image.pixsize[0])) ** 2. * e)
        amp = 4. * np.log(2) * flux / (np.pi * (bmaj/abs(image.pixsize[0]))**2 * e)

        # Create gaussian function of (x, y) with given parameters
        gaussf = gaussian(amp, x0, y0, bmaj, e, bpa=bpa)

        # Calculating angular distances of cells from center of component
        # from cell numbers to relative distances
        # arrays with elements from 1 to imsize
        x, y = np.mgrid[1: image.imsize[0] + 1,
                        1: image.imsize[1] + 1]
        # from -imsize/2 to imsize/2
        x = x - x_c
        y = y - y_c
        # the same in rads
        x = x * dx
        y = y * dy
        ## relative to component center
        #x = x - x0
        #y = y - y0
        ## convert to mas cause all params are in mas
        #x = x / mas_to_rad
        #y = y / mas_to_rad

        # Creating grid with component's flux at each cell
        fluxes = gaussf(x, y)

        if beam is not None:
            fluxes = beam.convolve(fluxes)

        # Adding component's flux to image grid
        image._image += np.rot90(fluxes)[::-1, ::]

    def substract_from_image(self, image, beam=None):
        """
        Substract component from given instance of ``Image`` class.
        """
        # Cellsize [rad]
        dx, dy = image.dx, image.dy
        pix_mas = abs(image.pixsize[0]/dx)
        # Center of image [pix]
        x_c, y_c = image.x_c, image.y_c

        # Parameters of component
        try:
            # Jy, mas, mas, mas,  , rad
            flux, x0, y0, bmaj, e, bpa = self._p
        # If we call method inside ``CGComponent``
        except ValueError:
            flux, x0, y0, bmaj = self._p
            e = 1.
            bpa = 0.

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad
        bmaj *= mas_to_rad

        # TODO: Is it [Jy/beam]??
        # Amplitude of gaussian component [Jy/beam]
        # amp = flux / (2. * math.pi * (bmaj / mas_to_rad) ** 2. * e)
        amp = flux / (2. * math.pi * (bmaj / abs(image.pixsize[0])) ** 2. * e)

        # Create gaussian function of (x, y) with given parameters
        gaussf = gaussian(amp, x0, y0, bmaj, e, bpa=bpa)

        # Calculating angular distances of cells from center of component
        # from cell numbers to relative distances
        # arrays with elements from 1 to imsize
        x, y = np.mgrid[1: image.imsize[0] + 1,
               1: image.imsize[1] + 1]
        # from -imsize/2 to imsize/2
        x = x - x_c
        y = y - y_c
        # the same in rads
        x = x * dx
        y = y * dy
        ## relative to component center
        #x = x - x0
        #y = y - y0
        ## convert to mas cause all params are in mas
        #x = x / mas_to_rad
        #y = y / mas_to_rad

        # Creating grid with component's flux at each cell
        fluxes = gaussf(x, y)

        if beam is not None:
            fluxes = beam.convolve(fluxes)

        # Substracting component's flux from image grid
        image._image -= fluxes


class CGComponent(EGComponent):
    """
    Class that implements circular gaussian component.
    """
    _parnames = ["flux", "x", "y", "bmaj"]
    size = len(_parnames)

    def __init__(self, flux, x, y, bmaj, fixed=None):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component phase center [mas].
        :param y:
            Y-coordinate of component phase center [mas].
        :param bmaj:
            Std of component size [mas].
        """
        super(CGComponent, self).__init__(flux, x, y, bmaj, e=1., bpa=0.,
                                          fixed=fixed)
        self._fixed = self._fixed[:-2]
        self._p = self._p[:-2]
        self.size = 4 - np.count_nonzero(self._fixed)

    def to_elliptic(self, e=1, bpa=0, fixed=None):
        """
        Create elliptical gaussian component from current circular.

        :param e: (optional)
            Eccentricity . (default: ``1``)
        :param bpa: (optional)
            Positional angle of the component major axis. (defualt: ``0``)
        :param fixed: (optional)
            Iterable of parameter's names that are fixed. If ``None`` use
            parameters of elliptical components.
        :return:
            Instance of ``EGcomponent``.
        """
        if fixed is None:
            fixed = np.concatenate((self._fixed, np.array([False, False])))
        return EGComponent(self.p[0], self.p[1], self.p[2], self.p[3], e, bpa,
                           fixed=np.array(EGComponent._parnames)[fixed])


class DeltaComponent(Component):
    """
    Class that implements delta-function component.
    """
    def __init__(self, flux, x, y, fixed=None):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component phase center [mas].
        :param y:
            Y-coordinate of component phase center [mas].
        """
        super(DeltaComponent, self).__init__()
        self._p = np.array([flux, x, y])
        self.size = 3
        if fixed is not None:
            for par in fixed:
                if par not in self._parnames:
                    raise Exception('Uknown parameter ' + str(par) + ' !')
                self._fixed[self._parnames.index(par)] = True
            self.size = 3 - np.count_nonzero(self._fixed)

    def to_circular(self, bmaj, fixed=None):
        """
        Create circular gaussian component from current delta.

        :param bmaj:
            Major axis size [mas].
        :param fixed: (optional)
            Iterable of parameter's names that are fixed. If ``None`` use
            parameters of elliptical components.
        :return:
            Instance of ``CGcomponent``.
        """
        if fixed is None:
            fixed = np.concatenate((self._fixed, np.array([False])))
        return CGComponent(self.p[0], self.p[1], self.p[2], bmaj,
                           fixed=np.array(CGComponent._parnames)[fixed])

    def to_elliptic(self, bmaj, e=1, bpa=0, fixed=None):
        """
        Create elliptical gaussian component from current delta.

        :param bmaj:
            Major axis size [mas].
        :param e: (optional)
            Eccentricity . (default: ``1``)
        :param bpa: (optional)
            Positional angle of the component major axis. (defualt: ``0``)
        :param fixed: (optional)
            Iterable of parameter's names that are fixed. If ``None`` use
            parameters of elliptical components.
        :return:
            Instance of ``EGcomponent``.
        """
        if fixed is None:
            fixed = np.concatenate((self._fixed,
                                    np.array([False, False, False])))
        return EGComponent(self.p[0], self.p[1], self.p[2], bmaj, e, bpa,
                           fixed=np.array(EGComponent._parnames)[fixed])

    def ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.
        """
        flux, x0, y0 = self._p

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad

        u = uv[:, 0]
        v = uv[:, 1]
        visibilities = (flux * np.exp(-2.0 * math.pi * 1j *
                                      (u[:, np.newaxis] * x0 +
                                       v[:, np.newaxis] * y0))).sum(axis=1)
        return visibilities

    def add_to_image(self, image, beam=None):
        """
        Add component to given instance of ``ImagePlane`` class.
        """
        dx, dy = image.dx, image.dy
        x_c, y_c = image.x_c, image.y_c

        flux, x0, y0 = self._p

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad

        x_coords = int(round(x0 / dx))
        y_coords = int(round(y0 / dy))
        # 2 means that x_c & x_coords should be zero-indexed actually both.
        x = x_c + x_coords - 2
        y = y_c + y_coords - 2
        # ``._image`` attribute contains model (FT of uv-data)
        # [y, x] - to get coincidence with fits clean maps

        if beam is not None:
            flux = beam.convolve(flux)

        try:
            image._image[y, x] += flux
        except IndexError:
            imsize = image.imsize[0]
            # print(x, y, flux)
            if x > imsize-1:
                x = imsize-1
            if x < 0:
                x = 0
            if y > imsize-1:
                y = imsize-1
            if y < 0:
                y = 0
            image._image[y, x] += flux

    def substract_from_image(self, image, beam=None):
        """
        Subtract component from given instance of ``ImagePlane`` class.
        """
        dx, dy = image.dx, image.dy
        x_c, y_c = image.x_c, image.y_c

        flux, x0, y0 = self._p

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad

        x_coords = int(round(x0 / dx))
        y_coords = int(round(y0 / dy))
        # 2 means that x_c & x_coords should be zero-indexed actually both.
        x = x_c + x_coords - 2
        y = y_c + y_coords - 2
        # ``._image`` attribute contains model (FT of uv-data)
        # [y, x] - to get coincidence with fits clean maps

        if beam is not None:
            flux = beam.convolve(flux)

        image._image[y, x] -= flux


# TODO: Add method of RM/alpha transformations? With arguments ``from_freq`` &
# ``to_freq``
class ImageComponent(Component):
    """
    Class that implements image component (2D-array of flux values).
    """
    def __init__(self, image, x, y):
        """
        :param image:
            2D numpy array with image.
        :param x:
            Iterable of zero axis coordinates.
        :param y:
            Iterable of first axis coordinates.
        """
        super(ImageComponent, self).__init__()
        self.imsize = np.shape(image)
        self.image = image
        self.x = x
        self.y = y
        self._parnames = [str(i) for i in range(self.imsize[0]*self.imsize[1])]
        self._fixed = np.zeros(len(self._parnames), dtype=bool)
        self._p = image.flatten()

    def add_to_image(self, image, beam=None):
        add = self.image
        if beam is not None:
            add = beam.convolve(self.image)
        image.image += np.rot90(add)[::-1, ::]

    def substract_from_image(self, image, beam=None):
        add = self.image
        if beam is not None:
            add = beam.convolve(self.image)
        image.image -= np.rot90(add)[::-1, ::]

    def ft(self, uv):
        u = uv[:, 0]
        v = uv[:, 1]
        return image_ft(self.image, self.x, self.y, u, v)


# TODO: Subclass ImageComponent
class ModelImageComponent(Component):
    """
    Class that represents model image that can be translated, rotated and
    scaled.
    """
    def __init__(self, image, x, y):
        """
        :param image:
            2D numpy array with image.
        :param x:
            Iterable of zero axis coordinates.
        :param y:
            Iterable of first axis coordinates.
        """
        super(ModelImageComponent, self).__init__()
        self.imsize = np.shape(image)
        self._image = image
        self.x = x
        self.y = y
        self._parnames.extend(['scale', 'theta'])
        self._fixed = np.concatenate((self._fixed,
                                      np.array([False, False]),))
        # Model hasn't any amplitude scaling, shift_x, shift_y, scale
        # or rotation (rad)
        self._p = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
        self.size = 5
        self._image_ft = None

    @property
    def image(self):
        """
        Returns original image rescaled, rotated, shifted and amplified.
        """
        print("Calculating image for parameters: {}".format(self.p))
        return transform_image(self._image, self.p[0], self.p[1], self.p[2],
                               self.p[3], self.p[4])

    # FIXME: I should inherit properties like image and ft and return
    # them transformed according to parameters

    def ft(self, uv):
        print("FT current image model")
        u = uv[:, 0]
        v = uv[:, 1]
        return image_ft(self.image.copy(), self.x, self.y, u, v)

    def add_to_image(self, image, beam=None):
        add = self.image
        if beam is not None:
            add = beam.convolve(self.image)
        image.image += np.rot90(add)[::-1, ::]

    def substract_from_image(self, image, beam=None):
        add = self.image
        if beam is not None:
            add = beam.convolve(self.image)
        image.image -= np.rot90(add)[::-1, ::]

