import numpy as np
import os
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
from math import copysign


class Profile():

    def __init__(self, coords, rigde_point):
        self._coords = coords
        self._cp = rigde_point
        self._data_dict = {}
        self._threshold = {}
        self._rel_dist = np.array([copysign(np.hypot(x-self._cp[0], y-self._cp[1]), y-self._cp[1]) for x, y in coords])
        self._fitspace = np.linspace(np.min(self._rel_dist), np.max(self._rel_dist), 1000)
        self._fit = None

    def _stk_checker(self, stk):
        if len(self._data_dict) == 0:
            raise Exception(f"No data found!")
        if stk is None:
            if len(self._data_dict) == 1:
                stk = list(self._data_dict.keys())[0]
            else:
                stk = "I"
        stk = stk.upper()
        if stk in self.stokes:
            return stk
        else:
            raise Exception(f"Unknown stokes! Avalible: {list(self._data_dict.keys())}")
            
    @property
    def stokes(self):
        return list(self._data_dict.keys())

    @property
    def coords(self):
        return self._coords

    @property
    def dec(self):
        return np.array([coord[1] for coord in self._coords])

    @property
    def ra(self):
        return np.array([coord[0] for coord in self._coords])

    @property
    def width(self):
        if self._fit is None:
            self._fit_gauss(self)
        return (np.max(self._fitspace[self._fit>np.max(self._fit)/2])-\
                np.min(self._fitspace[self._fit>np.max(self._fit)/2]))/2


    def get_dec_w_threshold(self, stk=None):
        stk = self._stk_checker(stk)
        return self.dec[self._data_dict[stk] > self._threshold[stk]]

    def get_ra_w_threshold(self, stk=None):
        stk = self._stk_checker(stk)
        return self.ra[self._data_dict[stk] > self._threshold[stk]]

    def load_data(self, profile, stk="I"):
        if len(profile) != len(self._coords):
            raise Exception("Loaded arr have incompatible dimentions!") 
        stk = stk.upper()
        self._threshold[stk] = 0.
        self._data_dict[stk] = np.array(profile)

    def set_threshold(self, trd, stk=None):
        stk = self._stk_checker(stk)
        self._threshold[stk] = trd

    def get(self, stk=None):
        stk = self._stk_checker(stk)
        return self._data_dict[stk][self._data_dict[stk] > self._threshold[stk]]

    def _fit_single_gauss(self, stk=None):
        stk = self._stk_checker(stk)

        def gausssian(x, a, x0, sigma): 
            return a*np.exp(-(x-x0)**2/(2*sigma**2)) 
        popt, pcov = curve_fit(gausssian, self._rel_dist, self._data_dict[stk])     
        gausssian_fit = gausssian(self._fitspace, popt[0], popt[1], popt[2])
        print()
        print(f'Gaussian fit paremeters for stokes {stk}:')
        print(f'Amplutude = {round(popt[0], 2)} mJy/beam')
        print(f'Dispersion = {abs(round(popt[2], 2))} mas')
        print(f'Max coordinate = {round(popt[1], 2)} mas')
        print()
        self._fit = gausssian_fit

    def _fit_gauss(self, stk=None):
        stk = self._stk_checker(stk)

        def gausssian_N(x, a, x0, sigma, N): 
            res = 0
            for a_, x0_, sigma_ in zip(a, x0, sigma):
                res += a_*np.exp(-(x-x0_)**2/(2*sigma_**2))
            return res

        def wrapper_gausssian_N(x, N, *args):
            a, b, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N])
            return gausssian_N(x, a, b, c, N)

        N = 1
        cond = np.inf
        popt2mem = None
        while N < 5:
            params_0 = np.array([1 for _ in range(N)]+[0 for i in range(N)]+[1 for _ in range(N)])
            popt, pcov = curve_fit(lambda x, *params_0: wrapper_gausssian_N(x, N, params_0), \
                    self._rel_dist, self._data_dict[stk], p0=params_0, method='trf')

            cond_ = np.linalg.cond(pcov)
            expected = wrapper_gausssian_N(self._rel_dist, N-1, popt)
            r = self._data_dict[stk] - expected
            chisq = np.sum((r/np.std(self._data_dict[stk]))**2)
            if chisq is None:
                break
            if chisq/N < cond:
                popt2mem = popt
                cond = chisq/N
            else:
                break
            N += 1

        gausssian_fit = wrapper_gausssian_N(self._fitspace, N-1, popt2mem)
        self._fit = gausssian_fit

    def plot(self, stk=None, outdir='', outfile='profile.png', fig=None, ax=None, 
             plot_fit=False, plot_profile=True, color=None, save=True):
        stk = self._stk_checker(stk)

        if fig is None:
            fig = plt.figure(figsize=(8.5, 6))
        if ax is None:
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel(r'Relative dist. (mas)')
        ax.set_ylabel(r'Flux, mJy/beam')
        print(self.coords[np.argmax(self._data_dict[stk])])
        if len(self.stokes) > 1:
            if color is None:
                color = list(np.random.choice(range(256), size=3)/256)
            ax.legend(loc='best')

        if plot_profile:
            ax.scatter(self._rel_dist[self._data_dict[stk]>self._threshold[stk]], 
                       self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]], 
                       color=color)

        if plot_fit:
            if self._fit is None:
                self._fit_gauss(stk=stk)
            gausssian_fit = self._fit
            ax.plot(self._fitspace[gausssian_fit>self._threshold[stk]], 
                    gausssian_fit[gausssian_fit>self._threshold[stk]], 
                    color=color, label=f'Stokes {stk}')

        if outfile is not None:
            plt.savefig(os.path.join(outdir, outfile), bbox_inches='tight')
            plt.close()
