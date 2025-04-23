import numpy as np
import os
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
from math import copysign
import warnings


def gausssian(x, a, x0, sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) 


class Profile():
    """
    Class that represents single jet tranverse profile.
    """
    def __init__(self, coords, rigde_point):
        self._coords = coords
        self._cp = rigde_point
        self._data_dict = {}
        self._threshold = {}
        self._rel_dist = np.array([copysign(np.hypot(x-self._cp[0], y-self._cp[1]), y-self._cp[1]) for x, y in coords])
        self._fitspace = np.linspace(np.min(self._rel_dist), np.max(self._rel_dist), 10000)
        self._fit = None
        self._fitparam = {}
        self._N_max = 4

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
        """
        Shorthand for all stokes.
        """
        return list(self._data_dict.keys())

    @property
    def coords(self):
        """
        Shorthand for profile points coordinates.
        """
        return self._coords

    @property
    def dec(self):
        """
        Shorthand for profile points declinations.
        """
        return np.array([coord[1] for coord in self._coords])

    @property
    def ra(self):
        """
        Shorthand for profile points right ascentions.
        """
        return np.array([coord[0] for coord in self._coords])

    @property
    def width(self):
        """
        Shorthand for profile width.
        """
        if self._fit is None:
            self._fit_gauss()
        return (np.max(self._fitspace[self._fit>np.max(self._fit)/2])-\
                np.min(self._fitspace[self._fit>np.max(self._fit)/2]))/2

    @property
    def N_max(self):
        """
        Shorthand maximum gusssians used in fit.
        """
        return self.N_max

    @N_max.setter
    def N_max(self, new):
        """
        Shorthand for setting maximum gusssians used in fit.
        """
        self._N_max = new

    def get_dec_w_threshold(self, stk=None):
        """
        Shorthand for profile points declinations. Points with flux below the 
        theshold are ignored.
        """
        stk = self._stk_checker(stk)
        return self.dec[self._data_dict[stk] > self._threshold[stk]]

    def get_ra_w_threshold(self, stk=None):
        """
        Shorthand for profile points right ascentions. Points with flux below the 
        theshold are ignored.
        """
        stk = self._stk_checker(stk)
        return self.ra[self._data_dict[stk] > self._threshold[stk]]

    def load_data(self, profile, stk="I"):
        """
        Load profile data.

        :param stk:
            Data points will be interpreted as observations in stokes stk. Default is 'I'.
        """
        if len(profile) != len(self._coords):
            raise Exception("Loaded arr have incompatible dimentions!") 
        stk = stk.upper()
        self._threshold[stk] = 0.
        self._data_dict[stk] = np.array(profile)

    def set_threshold(self, trd, stk=None):
        """
        Sets minumum flux level for a profile in stokes stk.
        """
        stk = self._stk_checker(stk)
        self._threshold[stk] = trd

    def get(self, stk=None):
        """
        Shorthand for getting profile data for stokes stk.
        """
        stk = self._stk_checker(stk)
        return self._data_dict[stk][self._data_dict[stk] > self._threshold[stk]]

    def _fit_single_gauss(self, stk=None):
        stk = self._stk_checker(stk)

        popt, pcov = curve_fit(gausssian, 
                               self._rel_dist[self._data_dict[stk]>self._threshold[stk]], 
                               self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]])     
        gausssian_fit = gausssian(self._fitspace, popt[0], popt[1], popt[2])
        print()
        print(f'Gaussian fit paremeters for stokes {stk}:')
        print(f'Amplutude = {round(popt[0], 2)} mJy/beam')
        print(f'Dispersion = {abs(round(popt[2], 2))} mas')
        print(f'Max coordinate = {round(popt[1], 2)} mas')
        print()
        self._fit = gausssian_fit
        self._fitparam["N"] = 1
        self._fitparam["popt"] = popt

    def _fit_gauss(self, stk=None):
        stk = self._stk_checker(stk)
        if self._threshold[stk] == 0.:
            warnings.warn("Threshold isnt set! Fit may be inaccurate.")
            std = np.std(self._data_dict[stk])
        else:
            # std = np.std(self._data_dict[stk][self._data_dict[stk]<self._threshold[stk]])
            std = 2*self._threshold[stk]

        def gausssian_N(x, a, x0, sigma, N): 
            res = 0.
            for a_, x0_, sigma_ in zip(a, x0, sigma):
                res += abs(a_)*np.exp(-(x-x0_)**2/(2*sigma_**2))
            return res

        def wrapper_gausssian_N(x, N, *args):
            a, b, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N])
            return gausssian_N(x, a, b, c, N)

        N = 1
        cond = np.inf
        popt2mem = None
        while N <= self._N_max:
            ma = np.max(self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]])
            diff = np.max(self._rel_dist[self._data_dict[stk]>self._threshold[stk]]) - \
                   np.min(self._rel_dist[self._data_dict[stk]>self._threshold[stk]])
            basex0 = diff/(N+1)
            params_0 = np.array([ma/2 for _ in range(N)]+ \
                                [-diff/2+basex0*(i+1) for i in range(N)]+ \
                                [basex0/2 for _ in range(N)])
            try: 
                popt, pcov = curve_fit(lambda x, *params_0: wrapper_gausssian_N(x, N, params_0), \
                                       self._rel_dist[self._data_dict[stk]>self._threshold[stk]], 
                                       self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]], 
                                       p0=params_0, method='dogbox')
            except RuntimeError:
                break
            expected = wrapper_gausssian_N(self._rel_dist[self._data_dict[stk]>self._threshold[stk]], N, popt)

            # # r = abs(self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]] - expected)
            # # chisq = np.sum((r/np.std(self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]]))**2)/N
            # obs = self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]][expected>0.]
            # expected = expected[expected>0.]
            # chisq = np.sum((obs-expected)**2/expected)/N
            # if chisq is None:
            #     break
            # if chisq < cond and np.max(wrapper_gausssian_N(self._fitspace, N, popt)) < ma*1.1:
            #     popt2mem = popt
            #     cond = chisq
            # else:
            #     break
            
            loglikelihood = 0.
            for obs, exp in zip(self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]], expected):
                loglikelihood += -((obs-exp)**2)/2/(std**2)
            Npoints = np.sum(self._data_dict[stk]>self._threshold[stk])
            bic = 2*3*N*(1+np.log(Npoints))-2*(loglikelihood-Npoints*np.log(1/np.sqrt(2*np.pi)/std))
            # print(f"N = {N}, bic = {bic}, logl = {2*loglikelihood}")
            if bic is None:
                break
            if bic < cond and np.max(wrapper_gausssian_N(self._fitspace, N, popt)) < ma*1.1:
                popt2mem = popt
                cond = bic
            else:
                break
            N += 1

        self._fitparam["N"] = 1
        if popt2mem is None:
            print("Fit unsuccessful!")
            return 0
        gausssian_fit = wrapper_gausssian_N(self._fitspace, N-1, popt2mem)
        self._fit = gausssian_fit
        self._fitparam["N"] = N-1
        self._fitparam["popt"] = popt2mem

    def plot(self, stk=None, outdir='', outfile='profile.png', fig=None, ax=None, 
             plot_fit=False, plot_profile=True, color=None, save=True):
        """
        Plot profile.

        :param stk:
            Stokes of profile to plot.
        :param outdir:
            Output directory.
        :param outfile:
            Name of file with the plot.
        :param plot_fit (optional):
            Plot profile fit? Default is False.
        :param plot_profile (optional):
            Plot profile itself? Default is True.
        :param color (optional):
            Plot color.
        :param save (optional):
           Wright the plot in a file? Default is True.
        """
        stk = self._stk_checker(stk)

        if fig is None:
            fig = plt.figure(figsize=(8.5, 6))
        if ax is None:
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel(r'Relative dist. (mas)')
        ax.set_ylabel(r'Flux, mJy/beam')

        if len(self.stokes) > 1:
            if color is None:
                color = list(np.random.choice(range(256), size=3)/256)
            ax.title(f'Stokes {stk}')

        if plot_profile:
            ax.scatter(self._rel_dist[self._data_dict[stk]>self._threshold[stk]], 
                       self._data_dict[stk][self._data_dict[stk]>self._threshold[stk]], 
                       color=color, label='data')

        if plot_fit:
            if self._fit is None:
                self._fit_gauss(stk=stk)
            gausssian_fit = self._fit
            if gausssian_fit is not None:
                ax.plot(self._fitspace[gausssian_fit>self._threshold[stk]], 
                        gausssian_fit[gausssian_fit>self._threshold[stk]], 
                        color=color, label=f'fit, N = {self._fitparam["N"]}')
                if self._fitparam["N"] > 1:
                    N = self._fitparam["N"]
                    args = self._fitparam["popt"]
                    a, b, c = list(args[:N]), list(args[N:2*N]), list(args[2*N:3*N])
                    for a_, x0, sigma in zip(a, b, c):
                        ax.plot(self._fitspace[gausssian_fit>self._threshold[stk]], 
                                gausssian(self._fitspace[gausssian_fit>self._threshold[stk]], abs(a_), x0, sigma), 
                                label=None)
                        
        ax.legend(loc="best")
        if outfile is not None:
            plt.savefig(os.path.join(outdir, outfile), bbox_inches='tight')
            plt.close()
