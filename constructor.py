import numpy as np
import os
import matplotlib.pyplot as plt
from image_local import CleanImage, Image, Jet_data, plot
from scipy.stats import circmean, circstd
from scipy.optimize import curve_fit 
from pycircstat2 import Circular
import pymc as pm
import pandas as pd
import arviz as az
import random as rnd
from utils_local import (find_bbox, find_image_std, mas_to_rad, degree_to_rad,
        normalize, circular_mean, normalize_angle, circular_dbscan,
        get_cluster_subarrays)
from profile import Profile


class Ridgeline_constructor(Jet_data):
    """
    Class that represents multiple stokes image with ridgeline on it.
    """
    def __init__(self):
        super().__init__()
        self._ridgeline = []
        self._bbox = None
        self._threshold = {}
        self._std = None
        self._stopping_criterion = ("std", 20)
        self._r_x_c = None
        self._r_y_c = None

    @property
    def ridgeline(self):
        """
        Shorthand for getting ridgeline data.

        :return:
        Array [[x1, y1], [x2, y2], ...] where each couple x_, y_ represents ra, dec of a sigle
        ridgline point (in mas).
        """
        if len(self._ridgeline) == 0:
            raise Exception(f"No ridgeline data found! Use .construct_ridge or .ridgeline_from_fits method")
        return self._ridgeline
    
    @ridgeline.setter
    def ridgeline(self, new_ridge):
        """
        Shorthand for setting ridgeline data.
        """
        self._ridgeline = new_ridge

    @property
    def length(self):
        """
        Shorthand for getting ridgeline length (in mas).
        """
        dist = 0.
        for idx in range(len(self._ridgeline)-1):
            dist += np.hypot(self._ridgeline[idx+1][0]-self._ridgeline[idx][0],
                             self._ridgeline[idx+1][1]-self._ridgeline[idx][1])
        return dist
    
    @property
    def stopping_criterion(self):

        return self._stopping_criterion
    
    @stopping_criterion.setter
    def stopping_criterion(self, new):

        self._stopping_criterion = new

    def threshold(self, stk):

        stk = stk.upper()
        if self._threshold[stk] is None:
            self._threshold[stk] = 20*self.get_std(stk)
        return self._threshold[stk]
    
    def set_threshold(self, stk, new):

        stk = stk.upper()
        self._threshold[stk] = new

    def get_std(self, stk):

        if self._std is None:  
            npixels_beam = np.pi*self.beam[0]*self.beam[1]/(4*np.log(2)*self.pixsize[1]**2)
            self._std = find_image_std(self.get_image(stk), beam_npixels=npixels_beam)
        return self._std

    def ridgeline_from_fits(self, fname):
        """
        Load ridgeline from provided fits file.
        """
        ridge = np.loadtxt(fname, comments='#')
        for point in ridge:
            self._ridgeline.append([point[0], point[1]])
        self._ridgeline = np.array(self._ridgeline)

    def ridgeline2file(self, fname):
        """
        Save ridgeline data to provided file.
        """
        outdata = {}
        outdata["ra (mas)"] = []
        outdata["dec (mas)"] = []
        for point in self.ridgeline:
            outdata["ra (mas)"].append(point[0])
            outdata["dec (mas)"].append(point[1])
        with open(fname, 'w') as fo:
            fo.write("! Ridgeline data points\n")
            fo.write("!")
            data = pd.DataFrame(outdata)
            fo.write(data.to_string(header=True, index=False, col_space=20))
            fo.close()

    def construct_ridge(self, stk='I', smoothing_factor=0.2, 
                        use_brightest_pixel_as_center=True, 
                        center=None, averaging_factor=1., 
                        core_factor=1.7, maxlen_cut=None):
        """
        Constructs ridgeline. 

        :param threshold:
            Threshold flux level. Pixels with lower flux are ignored. Default is 20*std.
        :param stk:
            Stokes of image used for ridgeline construction. Default is 'I'.
        :param smoothing_factor (optional):
            Spline smoothing factor. 
        """
        stk = stk.upper()
        img = self.get_image(stk)
        # self.x_c, self.y_c = np.unravel_index(img.argmax(), img.shape)
        if use_brightest_pixel_as_center:
            self._r_x_c, self._r_y_c = np.unravel_index(img.argmax(), img.shape)
        else:
            assert center is not None, "Use brightest pixel as center or provide one!"
            self._r_y_c, self._r_x_c = self._convert_coordinate(center)
            
        xy_c_mas = self._convert_array_coordinate((self._r_x_c, self._r_y_c))       
        core_radius = core_factor*self.beam[1]
        lmapsize = round(np.hypot(self.imsize[0], self.imsize[1]))
        lmap = [[[], [], []] for _ in range(lmapsize)]
        for x in np.arange(self.imsize[0]):
            for y in np.arange(self.imsize[1]):
                if not img[y, x] > self.threshold(stk):
                    continue
                length = np.hypot(x - self._r_x_c, y - self._r_y_c)
                if x - self._r_x_c != 0:
                    angle = -self.beam[2] + np.arctan2(y - self._r_y_c, x - self._r_x_c)
                else:
                    angle = -self.beam[2] + np.pi/2
                beam_r = 1/np.sqrt((np.sin(angle)/self.beam[1])**2+(np.cos(angle)/self.beam[0])**2)
                length = round(round(length/beam_r*self.beam[1]/averaging_factor)*averaging_factor)
                xy_mas = self._convert_array_coordinate((x, y))
                r = np.hypot(xy_mas[0]-xy_c_mas[0], xy_mas[1]-xy_c_mas[1])
                if r > 0:
                    lmap[length][0].append(r)
                    lmap[length][2].append(img[y, x])
                    if xy_mas[1]-xy_c_mas[1] <= 0:
                        lmap[length][1].append(np.pi - np.arcsin(-(-xy_mas[0]+xy_c_mas[0])/r))
                    else:
                        lmap[length][1].append(np.arcsin(-(-xy_mas[0]+xy_c_mas[0])/r))
                if length == 30:
                    plt.scatter([x], [y])

        ridgeline_polar = [[], [], []]
        ridgeline_polar[0].append(0)
        ridgeline_polar[1].append(0)
        ridgeline_polar[2].append(100)
        
        for length_arr in lmap:
            if len(length_arr[0]) > 2 and length_arr[0][0] > 0:

                length_arr = np.array(length_arr)
                mean = circular_mean(length_arr[1], w=normalize(length_arr[2]))
                if length_arr[1].max() < mean:
                    length_arr[1][length_arr[1] < 0] += 2*np.pi
                length_arr = length_arr[:, length_arr[1].argsort()]

                # length_arr = length_arr[:, length_arr[2]>np.max(length_arr[2])/5]

                # clustrerise angles array, find the main cluster
                labels = circular_dbscan(length_arr[1], eps=0.3, min_samples=5)
                subarrays = get_cluster_subarrays(length_arr[2], labels)
                peak = 0.
                peak_label = 0.
                for label in subarrays:
                    subarray = subarrays[label]
                    if np.max(subarray) > peak:
                        peak = np.max(subarray)
                        peak_label = label
                length_arr = np.array([get_cluster_subarrays(length_arr[0], labels)[peak_label],
                                        get_cluster_subarrays(length_arr[1], labels)[peak_label],
                                        subarrays[peak_label]])

                if self.stopping_criterion[0] == "std":
                    if np.max(length_arr[2]) < self.stopping_criterion[1]*self.get_std(stk):
                        continue
                                           
                    direction = circular_mean(length_arr[1], w=normalize(length_arr[2]))
                    # plt.plot(length_arr[1], length_arr[2])
                    # plt.savefig(f"{np.mean(length_arr[0])}_mas.png")
                    # plt.clf()
                    # direction = length_arr[1][np.argmax(length_arr[2])]

                elif self.stopping_criterion[0] == "Hovatta":
                    
                    # while np.argmax(length_arr[2]) < len(length_arr[1])/2:
                    #     length_arr[1, -1] -= 2*np.pi
                    #     length_arr = np.roll(length_arr, 1, axis=1)
                    # while np.argmax(length_arr[2]) > len(length_arr[1])/2:
                    #     length_arr[1, 0] += 2*np.pi
                    #     length_arr = np.roll(length_arr, -1, axis=1)
                    
                    # find proper cut in angle array (resolve 2pi discrepancy)
                    sorted_angles = length_arr[1]
                    n = len(sorted_angles)
                    shifted = np.tile(sorted_angles, (n, 1))
                    for i in range(n):
                        shifted[i, i+1:] -= 2 * np.pi
                    ranges = np.ptp(shifted, axis=1)  # ptp = peak to peak = max - min                    
                    best_idx = np.argmin(ranges)
                    length_arr[1] = shifted[best_idx]
                    length_arr = length_arr[:, length_arr[1].argsort()]

                    # in case of large angle ranges apply cut arr minimum
                    best_range = ranges[best_idx]
                    if best_range > np.pi:
                        arg_min = np.argmin(length_arr[2])
                        length_arr[1, arg_min+1:] -= 2 * np.pi
                        length_arr = length_arr[:, length_arr[1].argsort()]


                    mean = np.average(length_arr[1], weights=normalize(length_arr[2]))
                    # length_arr = length_arr[:, length_arr[1] - mean > -np.pi/2]
                    # length_arr = length_arr[:, length_arr[1] - mean < np.pi/2]
                    # mean = circular_mean(length_arr[1], w=normalize(length_arr[2]))
                        
                    P = Profile([(0, x) for x in length_arr[1]], (0, mean))
                    P.load_data(length_arr[2], stk=stk)
                    P.N_max = 1
                    P.set_threshold(self.stopping_criterion[1]/2*self.get_std(stk))
                    if len(P.get()) < 10:
                        continue
                    P._fit_single_gauss(stk=stk, initial_guess=(np.max(length_arr[2]), 
                                        0., (length_arr[1].max()-length_arr[1].min())/4))
                    # if length_arr[0, 0] > 1.55 and length_arr[0, 0] < 1.56:
                        # print(length_arr)
                    # P.plot(stk=stk, outfile=f"test_r={length_arr[0, 0]}.png", plot_fit=True)
                        # raise
                        
                    try:
                        amp = P.fitparam[0]
                        # direction = P.fitparam[1] + mean
                        direction = mean
                    except KeyError:
                        if length_arr[0][0] < core_radius:
                            amp = np.inf
                            direction = mean
                        else:
                            continue
                        # amp = length_arr[2].max()/2
                    # P.plot(stk=stk, outfile=f"test_r={length_arr[0, 0]}.png", plot_fit=True)
                    # print(self.stopping_criterion[1]*self.get_std(stk))
                    if amp < self.stopping_criterion[1]*self.get_std(stk) or \
                       length_arr[2].max() < self.stopping_criterion[1]*self.get_std(stk):
                        continue
                    # P.plot(stk=stk, outfile=f"test_r={length_arr[0, 0]}.png", plot_fit=True)
                else:
                    raise Exception("Unknown stopping criterion!")

                r_mean = np.mean(np.array(length_arr[0]))
                if len(ridgeline_polar[0]) > 5 and r_mean > 3*ridgeline_polar[0][-1]:
                    continue

                ridgeline_polar[0].append(r_mean)
                ridgeline_polar[1].append(direction)
                ridgeline_polar[2].append(1)

        ridgeline_polar = np.array(ridgeline_polar)

        # shift angles on 2pi
        mean = circmean(ridgeline_polar[1])
        std = circstd(ridgeline_polar[1])
        ridgeline_polar[1] = normalize_angle(ridgeline_polar[1])

        def circmedian(angs):
            pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
            pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
            pdists = np.abs(pdists).sum(1)
            return angs[np.argmin(pdists)]

        # delete too different points
        # get points around the core with mean angle
        for i in np.arange(ridgeline_polar[2].size - 2, -1, -1):
            if ridgeline_polar[0][i] == 0:
                ridgeline_polar[1][i] = ridgeline_polar[1][i + 1]
            # if np.abs(ridgeline_polar[1][i] - mean) > std:
            #     ridgeline_polar[1][i] = circmedian(ridgeline_polar[1, i+1:i+5])
            beam_r = 1/np.sqrt((np.cos(ridgeline_polar[1][i]+self.beam[2])/self.beam[1])**2+\
                    (np.sin(ridgeline_polar[1][i]+self.beam[2])/self.beam[0])**2)
            factor = beam_r/self.beam[1]
            if ridgeline_polar[0][i] < core_radius*factor:
                ridgeline_polar[1][i] = circmedian(ridgeline_polar[1, i+1:i+5])

        ridgeline_polar[1] = normalize_angle(ridgeline_polar[1])
        for i in np.arange(ridgeline_polar[1].size):
            if np.isnan(ridgeline_polar[1][i]):
                ridgeline_polar[1][i] = ridgeline_polar[1][i - 1]

        i = 0
        while i < len(ridgeline_polar[1])-1:
            angle = ridgeline_polar[1][i]
            if abs(ridgeline_polar[1][i-1] - angle) + abs(ridgeline_polar[1][i+1] - angle) > \
               2*abs(ridgeline_polar[1][i-1] - ridgeline_polar[1][i+1]):
                ridgeline_polar = np.delete(ridgeline_polar, i, axis=1)
            else:
                i+=1

        if maxlen_cut is not None:
            while np.max(ridgeline_polar[0]) > maxlen_cut:
                ridgeline_polar = ridgeline_polar[:, :-1]

        ridgeline_polar = ridgeline_polar[:, ridgeline_polar[0].argsort()]
        
        # fix larger swings in ridgeline
        i = 2
        while i < len(ridgeline_polar[0]):
            r = ridgeline_polar[0, i]
            factor = (r-ridgeline_polar[0, i-1])/averaging_factor/self.pixsize[1]*mas_to_rad
            tolerable_swing = np.deg2rad(100/r+5) * np.sqrt(factor)
            if abs(ridgeline_polar[1, i]-ridgeline_polar[1, i-1]) > tolerable_swing:
                if i != len(ridgeline_polar[0])-1:
                    ridgeline_polar = np.concatenate((ridgeline_polar[:, :i], ridgeline_polar[:, i+1:]), axis=1)
                else:
                    ridgeline_polar = ridgeline_polar[:, :i]
            else:
                i += 1

        # c = Circular(ridgeline_polar[1], w=None, alpha=0.05, unit="radian")
        # ridge_median = c.median
        # ridgeline_polar = ridgeline_polar[:, ridgeline_polar[1] < ridge_median+np.pi/6]
        # ridgeline_polar = ridgeline_polar[:, ridgeline_polar[1] > ridge_median-np.pi/6]

        # ridgeline_polar = ridgeline_polar[:, ridgeline_polar[0].argsort()]
        # while ridgeline_polar[1].max() - ridgeline_polar[1].min() > np.pi/3:
        #     ridgeline_polar = ridgeline_polar[:, :-1]
        from scipy.interpolate import UnivariateSpline
        maxlen_coord = np.max(ridgeline_polar[0])
        if len(ridgeline_polar[0]) < 4:
            print("Unable to construct ridgeline!")
            return 0

        spl = UnivariateSpline(list(ridgeline_polar[0])+[maxlen_coord*1.2], 
                               list(ridgeline_polar[1])+[circmean(ridgeline_polar[1])], 
                               w=list(ridgeline_polar[2])+[2.], 
                               s=smoothing_factor)
        rs = np.linspace(0, maxlen_coord, 1000)
        thetas = spl(rs)

        while thetas.max() - thetas.min() > np.pi/2:
            ridgeline_polar = ridgeline_polar[:, :-1]
            maxlen_coord = np.max(ridgeline_polar[0])
            spl = UnivariateSpline(list(ridgeline_polar[0])+[maxlen_coord*1.2], 
                                list(ridgeline_polar[1])+[circmean(ridgeline_polar[1])], 
                                w=list(ridgeline_polar[2])+[2.], 
                                s=smoothing_factor)
            rs = np.linspace(0, maxlen_coord, 1000)
            thetas = spl(rs)

        dec_c, ra_c = self._convert_array_coordinate((self._r_x_c, self._r_y_c))
        # dec_c, ra_c = 0., 0.

        # self._ridgeline = []
        # for r, theta in zip(ridgeline_polar[0], ridgeline_polar[1]):
        #     self._ridgeline.append([r*np.cos(theta)+ra_c, r*np.sin(theta)+dec_c])
        # self._ridgeline = np.array(self._ridgeline)
        
        self._ridgeline = []
        for r, theta in zip(rs, thetas):
            self._ridgeline.append([r*np.cos(theta)+ra_c, r*np.sin(theta)+dec_c])
        self._ridgeline = np.array(self._ridgeline)

    def plot(self, stk=None, outdir='', outfile='fig.png', fig=None, ax=None, min_abs_level=None,
             abs_levels=None, contour_color="black", ridge_size=None, ridge_color="grey", vectors=None,
             plot_ridge=True, corner_level_factor=3.):
        """
        Plot image.

        :param stk:
            Stokes of image to plot.
        :param outdir:
            Output directory.
        :param outfile:
            Name of file with the plot.
        """
        if stk is None:
            if len(self._image_dict) == 1:
                stk = list(self._image_dict.keys())[0]
            else:
                stk = "I"
        stk = stk.upper()
        img = self.get_image(stk)
        # npixels_beam = np.pi * self.beam[0] * self.beam[1] / (4 * np.log(2) * self.pixsize[1] ** 2)
        std = self.get_std(stk) # find_image_std(img, beam_npixels=npixels_beam)

        if min_abs_level is None:
            min_abs_level = 3 * std 
        # min_abs_level = 1e-5
        if self._bbox is None:
            min_abs_level_ = min_abs_level
            blc, trc = find_bbox(img[:, ::-1], level=min_abs_level_*corner_level_factor, min_maxintensity_mjyperbeam=10*std,
                                min_area_pix=0., delta=10)
            if blc[0] == 0: blc = (blc[0] + 1, blc[1])
            if blc[1] == 0: blc = (blc[0], blc[1] + 1)
            if trc[0] == img.shape[0]: trc = (trc[0] - 1, trc[1])
            if trc[1] == img.shape[1]: trc = (trc[0], trc[1] - 1)
            self._bbox = blc, trc
            while blc == (1, 1) and trc == (img.shape[0]-1, img.shape[1]-1):
                min_abs_level_ *= 1.5
                if min_abs_level_ > np.max(img) or min_abs_level_ == 0:
                    min_abs_level_ = np.max(img)/1000
                    break
                blc, trc = find_bbox(img[:, ::-1], level=min_abs_level_*corner_level_factor, min_maxintensity_mjyperbeam=10*std,
                                    min_area_pix=0., delta=10)
                if blc[0] == 0: blc = (blc[0] + 1, blc[1])
                if blc[1] == 0: blc = (blc[0], blc[1] + 1)
                if trc[0] == img.shape[0]: trc = (trc[0] - 1, trc[1])
                if trc[1] == img.shape[1]: trc = (trc[0], trc[1] - 1)
                self._bbox = blc, trc
        else:
            blc, trc = self._bbox

        label_size = 16
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        plt.rcParams['axes.titlesize'] = label_size
        plt.rcParams['axes.labelsize'] = label_size
        plt.rcParams['font.size'] = label_size
        plt.rcParams['legend.fontsize'] = label_size
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        if fig is None:
            fig = plt.figure(figsize=(8.5, 6))
        if ax is None:
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel(r'Relative R.A. (mas)')
        ax.set_ylabel(r'Relative Decl. (mas)')
        if plot_ridge:
            if len(self._ridgeline) > 0:
                ax.scatter(self._ridgeline[:, 0], self._ridgeline[:, 1], s=ridge_size, color=ridge_color)
            else:
                print("No ridgeline to plot! Run Ridgeline_constructor.construct_ridge in order to build one!")
        if vectors is None:
            vectors_mask = None
        else:
            vectors_mask = img[:, ::-1] < min_abs_level
        plot(contours=img[:, ::-1],  # subtract_gaussian_core(image_data_i, mapsize, 40*std),
                colors=None, colors_mask=None,
                vectors=vectors, vectors_mask=vectors_mask,
                x=self.x, show_beam=True, k=2, vinc=4, cmap='Oranges',
                y=self.y[::-1], min_abs_level=min_abs_level, abs_levels=abs_levels,
                blc=blc, trc=trc, close=False, contour_color=contour_color, plot_colorbar=False,
                beam=self.beam, fig=fig, axes=ax, label_size=label_size, colorbar_label=None,
                vector_scale=4)
        if outfile is not None:
            fig.savefig(os.path.join(outdir, outfile), bbox_inches='tight')


class Gauss_process_ridgeline(Ridgeline_constructor):

    def __init__(self):
        super().__init__()
        self._gaussian_ridgelines = []

    @property
    def gaussian_ridgelines(self):
        return self._gaussian_ridgelines

    def construct_ridge(self, stk='I', smoothing_factor=0.2, 
                        use_brightest_pixel_as_center=True, 
                        center=None, averaging_factor=1.,
                        N=100):
        """
        Constructs ridgeline. 

        :param threshold:
            Threshold flux level. Pixels with lower flux are ignored. Default is 20*std.
        :param stk:
            Stokes of image used for ridgeline construction. Default is 'I'.
        :param smoothing_factor (optional):
            Spline smoothing factor. 
        """
        # stk = stk.upper()
        # img = self.get_image(stk)
        # # self.x_c, self.y_c = np.unravel_index(img.argmax(), img.shape)
        # if use_brightest_pixel_as_center:
        #     self._r_x_c, self._r_y_c = np.unravel_index(img.argmax(), img.shape)
        # else:
        #     assert center is not None, "Use brightest pixel as center or provide one!"
        #     self._r_y_c, self._r_x_c = self._convert_coordinate(center)
            
        # xy_c_mas = self._convert_array_coordinate((self._r_x_c, self._r_y_c))       
        # core_radius = 1.7*self.beam[1]
        # lmapsize = round(np.hypot(self.imsize[0], self.imsize[1]))
        # lmap = [[[], [], []] for _ in range(lmapsize)]
        # for x in np.arange(self.imsize[0]):
        #     for y in np.arange(self.imsize[1]):
        #         if not img[y, x] > self.threshold(stk):
        #             continue
        #         length = np.hypot(x - self._r_x_c, y - self._r_y_c)
        #         if x - self._r_x_c != 0:
        #             angle = -self.beam[2] + np.arctan2(y - self._r_y_c, x - self._r_x_c)
        #         else:
        #             angle = -self.beam[2] + np.pi/2
        #         beam_r = 1/np.sqrt((np.sin(angle)/self.beam[1])**2+(np.cos(angle)/self.beam[0])**2)
        #         length = round(round(length/beam_r*self.beam[1]/averaging_factor)*averaging_factor)
        #         xy_mas = self._convert_array_coordinate((x, y))
        #         r = np.hypot(xy_mas[0]-xy_c_mas[0], xy_mas[1]-xy_c_mas[1])
        #         if r > 0:
        #             lmap[length][0].append(r)
        #             lmap[length][2].append(img[y, x])
        #             if xy_mas[1]-xy_c_mas[1] <= 0:
        #                 lmap[length][1].append(np.pi - np.arcsin(-(-xy_mas[0]+xy_c_mas[0])/r))
        #             else:
        #                 lmap[length][1].append(np.arcsin(-(-xy_mas[0]+xy_c_mas[0])/r))
        #         if length == 30:
        #             plt.scatter([x], [y])

        # ridgeline_polar = [[], [], []]
        # ridgeline_polar[0].append(0)
        # ridgeline_polar[1].append(0)
        # ridgeline_polar[2].append(100)
        
        # for length_arr in lmap:
        #     if len(length_arr[0]) > 2 and length_arr[0][0] > 0:
        #         if self.stopping_criterion[0] == "std":
        #             if np.max(length_arr[2]) < self.stopping_criterion[1]*self.get_std(stk):
        #                 continue
        #             direction = circular_mean(length_arr[1], w=normalize(length_arr[2]))
        #             # direction = length_arr[1][np.argmax(length_arr[2])]
        #         elif self.stopping_criterion[0] == "Hovatta":
        #             length_arr = np.array(length_arr)
        #             mean = circular_mean(length_arr[1], w=normalize(length_arr[2]))
        #             if length_arr[1].max() < mean:
        #                 length_arr[1][length_arr[1] < 0] += 2*np.pi
        #             length_arr = length_arr[:, length_arr[1].argsort()]

        #             labels = circular_dbscan(length_arr[1], eps=0.3, min_samples=5)
        #             subarrays = get_cluster_subarrays(length_arr[2], labels)
        #             peak = 0.
        #             peak_label = 0.
        #             for label in subarrays:
        #                 subarray = subarrays[label]
        #                 if np.max(subarray) > peak:
        #                     peak = np.max(subarray)
        #                     peak_label = label
        #             length_arr = np.array([get_cluster_subarrays(length_arr[0], labels)[peak_label],
        #                                    get_cluster_subarrays(length_arr[1], labels)[peak_label],
        #                                    subarrays[peak_label]])
        #             # while np.argmax(length_arr[2]) < len(length_arr[1])/2:
        #             #     length_arr[1, -1] -= 2*np.pi
        #             #     length_arr = np.roll(length_arr, 1, axis=1)
        #             # while np.argmax(length_arr[2]) > len(length_arr[1])/2:
        #             #     length_arr[1, 0] += 2*np.pi
        #             #     length_arr = np.roll(length_arr, -1, axis=1)
                    
        #             sorted_angles = length_arr[1]
        #             n = len(sorted_angles)
        #             shifted = np.tile(sorted_angles, (n, 1))
        #             for i in range(n):
        #                 shifted[i, i+1:] -= 2 * np.pi
        #             ranges = np.ptp(shifted, axis=1)  # ptp = peak to peak = max - min                    
        #             best_idx = np.argmin(ranges)
        #             length_arr[1] = shifted[best_idx]
        #             length_arr = length_arr[:, length_arr[1].argsort()]

        #             mean = np.average(length_arr[1], weights=normalize(length_arr[2]))
        #             # length_arr = length_arr[:, length_arr[1] - mean > -np.pi/2]
        #             # length_arr = length_arr[:, length_arr[1] - mean < np.pi/2]
        #             # mean = circular_mean(length_arr[1], w=normalize(length_arr[2]))
                        
        #             P = Profile([(0, x) for x in length_arr[1]], (0, mean))
        #             P.load_data(length_arr[2], stk=stk)
        #             P.N_max = 1
        #             P.set_threshold(self.stopping_criterion[1]/2*self.get_std(stk))
        #             if len(P.get()) < 10:
        #                 continue
        #             P._fit_single_gauss(stk=stk, initial_guess=(np.max(length_arr[2]), 
        #                                 0., (length_arr[1].max()-length_arr[1].min())/4))
        #             # if length_arr[0, 0] > 1.55 and length_arr[0, 0] < 1.56:
        #                 # print(length_arr)
        #             # P.plot(stk=stk, outfile=f"test_r={length_arr[0, 0]}.png", plot_fit=True)
        #                 # raise
                        
        #             try:
        #                 amp = P.fitparam[0]
        #                 # direction = P.fitparam[1] + mean
        #                 direction = mean
        #             except KeyError:
        #                 if length_arr[0][0] < core_radius:
        #                     amp = np.inf
        #                     direction = mean
        #                 else:
        #                     continue
        #                 # amp = length_arr[2].max()/2
        #             # P.plot(stk=stk, outfile=f"test_r={length_arr[0, 0]}.png", plot_fit=True)
        #             # print(self.stopping_criterion[1]*self.get_std(stk))
        #             if amp < self.stopping_criterion[1]*self.get_std(stk) or \
        #                length_arr[2].max() < self.stopping_criterion[1]*self.get_std(stk):
        #                 continue
        #             # P.plot(stk=stk, outfile=f"test_r={length_arr[0, 0]}.png", plot_fit=True)
        #         else:
        #             raise Exception("Unknown stopping criterion!")

        #         r_mean = np.mean(np.array(length_arr[0]))
        #         if len(ridgeline_polar[0]) > 5 and r_mean > 3*ridgeline_polar[0][-1]:
        #             continue

        #         ridgeline_polar[0].append(r_mean)
        #         ridgeline_polar[1].append(direction)
        #         ridgeline_polar[2].append(1)

        # ridgeline_polar = np.array(ridgeline_polar)

        # # shift angles on 2pi
        # mean = circmean(ridgeline_polar[1])
        # std = circstd(ridgeline_polar[1])
        # ridgeline_polar[1] = normalize_angle(ridgeline_polar[1])

        # def circmedian(angs):
        #     pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
        #     pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
        #     pdists = np.abs(pdists).sum(1)
        #     return angs[np.argmin(pdists)]

        # # delete too different points
        # # get points around the core with mean angle
        # for i in np.arange(ridgeline_polar[2].size - 2, -1, -1):
        #     if ridgeline_polar[0][i] == 0:
        #         ridgeline_polar[1][i] = ridgeline_polar[1][i + 1]
        #     # if np.abs(ridgeline_polar[1][i] - mean) > std:
        #     #     ridgeline_polar[1][i] = circmedian(ridgeline_polar[1, i+1:i+5])
        #     beam_r = 1/np.sqrt((np.cos(ridgeline_polar[1][i]+self.beam[2])/self.beam[1])**2+\
        #             (np.sin(ridgeline_polar[1][i]+self.beam[2])/self.beam[0])**2)
        #     factor = beam_r/self.beam[1]
        #     if ridgeline_polar[0][i] < core_radius*factor:
        #         ridgeline_polar[1][i] = circmedian(ridgeline_polar[1, i+1:i+5])

        # ridgeline_polar[1] = normalize_angle(ridgeline_polar[1])
        # for i in np.arange(ridgeline_polar[1].size):
        #     if np.isnan(ridgeline_polar[1][i]):
        #         ridgeline_polar[1][i] = ridgeline_polar[1][i - 1]

        # i = 0
        # while i < len(ridgeline_polar[1])-1:
        #     angle = ridgeline_polar[1][i]
        #     if abs(ridgeline_polar[1][i-1] - angle) + abs(ridgeline_polar[1][i+1] - angle) > \
        #        2*abs(ridgeline_polar[1][i-1] - ridgeline_polar[1][i+1]):
        #         ridgeline_polar = np.delete(ridgeline_polar, i, axis=1)
        #     else:
        #         i+=1

        # ridgeline_polar = ridgeline_polar[:, ridgeline_polar[0].argsort()]
        # while ridgeline_polar[1].max() - ridgeline_polar[1].min() > np.pi/3:
        #     ridgeline_polar = ridgeline_polar[:, :-1]
        # from scipy.interpolate import UnivariateSpline
        # maxlen_coord = np.max(ridgeline_polar[0])
        # if len(ridgeline_polar[0]) < 4:
        #     print("Unable to construct ridgeline!")
        #     return 0

        # spl = UnivariateSpline(list(ridgeline_polar[0])+[maxlen_coord*1.2], 
        #                        list(ridgeline_polar[1])+[circmean(ridgeline_polar[1])], 
        #                        w=list(ridgeline_polar[2])+[2.], 
        #                        s=smoothing_factor)
        # rs = np.linspace(0, maxlen_coord, 1000)
        # thetas = spl(rs)

        # while thetas.max() - thetas.min() > np.pi/3:
        #     ridgeline_polar = ridgeline_polar[:, :-1]
        #     maxlen_coord = np.max(ridgeline_polar[0])
        #     spl = UnivariateSpline(list(ridgeline_polar[0])+[maxlen_coord*1.2], 
        #                         list(ridgeline_polar[1])+[circmean(ridgeline_polar[1])], 
        #                         w=list(ridgeline_polar[2])+[2.], 
        #                         s=smoothing_factor)
        #     rs = np.linspace(0, maxlen_coord, 100)
        #     thetas = spl(rs)

        # ell_true = maxlen_coord/2
        super().construct_ridge(stk=stk, smoothing_factor=smoothing_factor, 
                        use_brightest_pixel_as_center=use_brightest_pixel_as_center, 
                        center=center, averaging_factor=averaging_factor)
    
        rs = []
        thetas = []
        dec_c, ra_c = self._convert_array_coordinate((self._r_x_c, self._r_y_c))
        for point in self.ridgeline:
            rs.append(np.hypot(point[0]-ra_c, point[1]-dec_c))
            thetas.append(np.arctan2(point[1]-dec_c, point[0]-ra_c))
            
        rs = np.array(rs)
        thetas = np.array(thetas)
        thetas[0] = thetas[1]

        maxlen_coord = rs[-1]
        ell_true = self.beam[1]
        # eta_true = np.pi/100 np.std(thetas)/2
        eta_true = np.std(thetas)/4 + np.arctan2(self.beam[1], maxlen_coord)/4
        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ell_true)
        scale = cov_func(rs[:, None]).eval()
        # Add jitter (1e-6) to the diagonal to guarantee positive definiteness
        scale += np.eye(len(rs)) * 1e-7
        mean_func = pm.gp.mean.Zero()
        tp_samples = pm.draw(pm.MvStudentT.dist(mu=mean_func(rs[:, None]).eval(),
                            scale=scale, nu=3), N)
        
        # tp_samples = pm.draw(pm.MvStudentT.dist(mu=mean_func(rs[:, None]).eval(), 
        #                      scale=cov_func(rs[:, None]).eval(), nu=3), N)


        # dec_c, ra_c = 0., 0.

        # self._ridgeline = []
        # for r, theta in zip(ridgeline_polar[0], ridgeline_polar[1]):
        #     self._ridgeline.append([r*np.cos(theta)+ra_c, r*np.sin(theta)+dec_c])
        # self._ridgeline = np.array(self._ridgeline)

        # self._ridgeline = []
        # for r, theta in zip(rs, thetas):
        #     self._ridgeline.append([r*np.cos(theta)+ra_c, r*np.sin(theta)+dec_c])
        # self._ridgeline = np.array(self._ridgeline)

        for sample in tp_samples:
            g_ridgeline = []
            for r, s, theta in zip(rs, sample, thetas):
                g_ridgeline.append([r*np.cos(theta+s)+ra_c, r*np.sin(theta+s)+dec_c])
            g_ridgeline = np.array(g_ridgeline)
            self._gaussian_ridgelines.append(g_ridgeline)

    def plot(self, stk=None, outdir='', outfile='fig.png', fig=None, ax=None, min_abs_level=None,
             abs_levels=None, contour_color="black", ridge_size=None, ridge_color="grey", vectors=None,
             plot_true_ridge=True, plot_gaussian_ridge=False, gaussian_ridge_color="red"):

        if fig is None:
            fig = plt.figure(figsize=(8.5, 6))
        if ax is None:
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        
        super().plot(stk=stk, outdir=None, outfile=None, fig=fig, ax=ax, min_abs_level=min_abs_level,
             abs_levels=abs_levels, contour_color=contour_color, ridge_size=ridge_size, ridge_color=ridge_color, 
             vectors=vectors, plot_ridge=plot_true_ridge)
        
        if type(plot_gaussian_ridge) == int:
            idx = plot_gaussian_ridge
            ax.scatter(self._gaussian_ridgelines[idx][:, 0], self._gaussian_ridgelines[idx][:, 1], s=ridge_size, 
                       color=gaussian_ridge_color)
        elif type(plot_gaussian_ridge) == bool and plot_gaussian_ridge == True:
            idx = rnd.randint(0, len(self._gaussian_ridgelines)-1)
            ax.scatter(self._gaussian_ridgelines[idx][:, 0], self._gaussian_ridgelines[idx][:, 1], s=ridge_size, 
                        color=gaussian_ridge_color)
        else: pass
        if outfile is not None:
            fig.savefig(os.path.join(outdir, outfile), bbox_inches='tight')


class Profile_constructor(Ridgeline_constructor):
    """
    Class that represents multiple stokes image with ridgeline on it and provides some methods
    for profile construction.
    """
    def __init__(self):
        super().__init__()

    def profile_from_idx(self, idx):
        """
        Provides profile from ridgeline point index.
        """
        if len(self._ridgeline) == 0:
            if len(self.stokes) == 1:
                warnings.warn("No rigeline found, constructing")
                self.construct_ridge(self, stk=self.stokes[0], smoothing_factor=0.2)
            elif "I" in self.stokes:
                warnings.warn("No rigeline found, constructing")
                self.construct_ridge(self, stk="I", smoothing_factor=0.2)
            else:
                raise Exception("No rigeline found, unable to construct!")
        assert (idx != 0 and idx != len(self._ridgeline)-2), "Unable to construct the profile!"
    
        dy = self._ridgeline[idx+1][1]-self._ridgeline[idx-1][1]
        dx = self._ridgeline[idx+1][0]-self._ridgeline[idx-1][0]
        ridge_direction = -np.arctan2(dy, dx)
        slope = np.tan(ridge_direction + np.pi/2)
        ridgeline_pix = self._convert_coordinate(self._ridgeline[idx])
        pix1 = self._in_img((-1., -1.), ridgeline_pix, slope)
        pix2 = self._in_img(self.imsize, ridgeline_pix, slope)
        sl = self.slice(pix1=pix1, pix2=pix2)
        coords = []
        for ra, dec in zip(sl["ra"], sl["dec"]):
            coords.append((ra, dec))
        P = Profile(coords, self._ridgeline[idx])
        for stk in self.stokes:
            P.load_data(sl[stk], stk=stk)

        return P

    def profile_from_distance(self, target_dist):
        """
        Provides profile on set distance along ridgeline.
        """
        dist = 0.0
        idx = 0 
        while target_dist > dist:
            if idx == len(self._ridgeline)-1:
                raise Exception(f"Set distance is too high, ridgeline only extends up to {round(dist, 1)} mas")
            dist += np.hypot(self._ridgeline[idx+1][0]-self._ridgeline[idx][0],
                             self._ridgeline[idx+1][1]-self._ridgeline[idx][1])
            idx += 1
        return self.profile_from_idx(idx)

    def _fit_profile_into_bbox(self, profile, stk=None):
        assert isinstance(profile, Profile), \
                        "variable profile must be of Profile object type!"
        if self._bbox is None:
            raise Exception("Unable to find bbox!")
        blc, trc = self._bbox
        ras, decs = [], []
        for ra, dec in zip(profile.get_ra_w_threshold(stk=stk), profile.get_dec_w_threshold(stk=stk)):
            y, x = self._convert_coordinate((ra, dec))
            if x <= blc[1] or x >= trc[1]:
                continue
            if y <= blc[0] or y >= trc[0]:
                continue
            ras.append(ra)
            decs.append(dec)
        return ras, decs

    def plot(self, stk=None, outdir='', outfile='fig.png', fig=None, ax=None, profile_to_plot=None, 
             min_abs_level=None, abs_levels=None, contour_color="black", ridge_size=None, ridge_color="grey",
             vectors=None):
        """
        Plot image.

        :param stk:
            Stokes of image to plot.
        :param outdir:
            Output directory.
        :param outfile:
            Name of file with the plot.
        :param profile_to_plot:
            List of profiles to plot. Each profile must be an instance of Profile.
        """
        if fig is None:
            fig = plt.figure(figsize=(8.5, 6))
        if ax is None:
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        super().plot(stk=stk, outdir=None, outfile=None, fig=fig, ax=ax, min_abs_level=min_abs_level,
                     abs_levels=abs_levels, contour_color=contour_color, ridge_size=ridge_size, 
                     ridge_color=ridge_color, vectors=vectors)
        if profile_to_plot is not None:
            if type(profile_to_plot) == list:
                for p in profile_to_plot:
                    assert isinstance(p, Profile), \
                        "Variable profiles_to_plot must contain Profile object or list of those!"
                    ra, dec = self._fit_profile_into_bbox(p, stk=stk)
                    ax.plot([ra[0], ra[-1]], [dec[0], dec[-1]])
            elif isinstance(profile_to_plot, Profile):
                ra, dec = self._fit_profile_into_bbox(profile_to_plot, stk=stk)
                ax.plot([ra[0], ra[-1]], [dec[0], dec[-1]])
            else:
                raise Exception("Variable profiles_to_plot must contain Profile object or list of those!")

        if outfile is not None:
            fig.savefig(os.path.join(outdir, outfile), bbox_inches='tight')
            plt.close()

        
if __name__ == "__main__":
    ccimage = "/home/rtodorov/jet-profile-construction/example/1652+398.u.stacked.icc.fits"
    ridge_file = "/home/rtodorov/jet-profile-construction/example/1652+398.u.stacked.icc.fits.ridge_ascii"

    r = Gauss_process_ridgeline()
    r.from_fits(ccimage, set_stokes='I')
    npixels_beam = np.pi * r.beam[0] * r.beam[1] / (4 * np.log(2) * r.pixsize[1] ** 2)
    std = find_image_std(r.get_image(stk='I'), beam_npixels=npixels_beam)
    r.set_threshold("I", 20*std)
    r.construct_ridge(stk='I', smoothing_factor=0.2)
    # r._bbox = ((200, 200), (300, 300))
    r.plot(stk='I', outfile='example/map_spoiled.png', plot_gaussian_ridge=True)

    raise
    # initialize profile constructor
    r = Profile_constructor()
    # load data from fits 
    # set_stokes parameter forcibly sets data stokes, otherwise stokes will be read from fits directly
    r.from_fits(ccimage, set_stokes='I')

    # construct ridgeline
    npixels_beam = np.pi * r.beam[0] * r.beam[1] / (4 * np.log(2) * r.pixsize[1] ** 2)
    std = find_image_std(r.get_image(stk='I'), beam_npixels=npixels_beam)
    r.set_threshold("I", 20*std)
    r.construct_ridge(stk='I', smoothing_factor=0.2)
    # ridgeline data also can be read from fits: r.ridgeline_from_fits(ridge_file)
    
    # get profiles from distance along ridgeline
    p = r.profile_from_distance(7)
    b = r.profile_from_distance(5)
    # profiles also can be constructed from redgeline point index: r.profile_from_idx(200)

    # setting lower flux level
    p.set_threshold(10*std)
    b.set_threshold(10*std)

    # individual profiles can be plot
    b.plot(outfile='example/profile.png', plot_fit=True)

    # one can get profiles width
    print(f"Width = {b.width} mas")

    # one can plot whole map with profiles on it
    r.plot(stk='I', profile_to_plot=[b, p], outfile='example/map.png')
