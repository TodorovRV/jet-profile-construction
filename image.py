import os
import math
import numpy as np
import astropy.io.fits as pf
import warnings
from scipy import signal
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from utils import (create_grid, create_mask, mas_to_rad, v_round,
                   get_fits_image_info_from_hdulist, get_hdu_from_hdulist)
from model import Model
from beam import CleanBeam
from skimage.registration import phase_cross_correlation
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib import cm

try:
    import pylab
except ImportError:
    pylab = None


# FIXME: This finds only dr that minimize std for shift - radius dependence
# TODO: use iterables of shifts and sizes as arguments. UNIX-way:)
def find_shift(image1, image2, max_shift, shift_step, min_shift=0,
               min_mask_r=0, max_mask_r=100, mask_step=1, upsample_factor=100):
    """
    Find shift between two images using our heuristic.

    :param image1:
        Instance of ``BasicImage`` class.
    :param image2:
        Instance of ``BasicImage`` class.
    :param max_shift:
        Maximum size of shift to check [pxl].
    :param shift_step:
        size of shift changes step [pxl].
    :param min_shift: (optional)
        Minimum size of shift to check [pxl]. (default: ``0``)
    :param max_mask_r: (optional)
        Maximum size of mask to apply. (default: ``100.``)
    :param mask_step: (optional)
        Size of mask size changes step. (default: ``5``)
    :return:
        Array of shifts.
    """
    shift_dict = dict()

    # Iterating over difference of mask sizes
    for dr in np.arange(min_shift, max_shift, shift_step):
        print("Using dr = {}".format(dr))
        shift_dict[dr] = list()
        print(shift_dict)
        # Iterating over mask sizes
        for r in range(min_mask_r, max_mask_r, mask_step):
            print("Using r = {}".format(r))
            r1 = r
            r2 = r + dr
            shift = image1.cross_correlate(image2,
                                           region1=(image1.x_c, image1.y_c, r1,
                                                    None),
                                           region2=(image2.x_c, image2.y_c, r2,
                                                    None),
                                           upsample_factor=100)
            print("Found shift = {}".format(shift))
            shift_dict[dr].append(shift)

    shift_value_dict = dict()
    for key, value in shift_dict.items():
        value = np.array(value)
        shifts = np.sqrt(value[:, 0] ** 2. + value[:, 1] ** 2.)
        shift_value_dict.update({key: np.std(shifts)})
    shift_values_dict = dict()
    for key, value in shift_dict.items():
        value = np.array(value)
        shifts = np.sqrt(value[:, 0] ** 2. + value[:, 1] ** 2.)
        shift_values_dict.update({key: shifts})

    # Searching for mask size difference that has minimal std in shifts
    # calculated for different mask sizes
    dr_tgt = sorted(shift_dict, key=lambda _: shift_value_dict[_])[0]
    # Shift values vs mask radius dependence for best mask size difference
    shift_values = shift_values_dict[dr_tgt]
    # Looking for first minimum
    idx = (np.diff(np.sign(np.diff(shift_values))) > 0).nonzero()[0] + 1
    # return shift_dict[dr_tgt][idx[0]]
    print("best at r={}, dr={}".format(idx[0], dr_tgt))
    shift = image1.cross_correlate(image2,
                                   region1=(image1.x_c, image1.y_c, idx[0],
                                            None),
                                   region2=(image2.x_c, image2.y_c, idx[0] + dr_tgt,
                                            None),
                                   upsample_factor=upsample_factor)
    return shift, shift_dict, shift_values_dict


# def find_bbox(array, level, delta=0.):
#     """
#     Find bounding box for part of image containing source.
#
#     :param array:
#         Numpy 2D array with image.
#     :param level:
#         Level at which threshold image in image units.
#     :param delta:
#         Extra space to add symmetrically [pixels].
#     :return:
#         Tuples of BLC & TRC.
#     """
#     from scipy.ndimage.measurements import label
#     from scipy.ndimage.morphology import generate_binary_structure
#     from skimage.measure import regionprops
#
#     signal = array > level
#     s = generate_binary_structure(2, 2)
#     labeled_array, num_features = label(signal, structure=s)
#     props = regionprops(labeled_array, intensity_image=array)
#
#     max_prop = props[0]
#     for prop in props[1:]:
#         if prop.max_intensity > max_prop.max_intensity:
#             max_prop = prop
#
#     bbox = max_prop.bbox
#     blc = (int(bbox[1]), int(bbox[0]))
#     trc = (int(bbox[3]), int(bbox[2]))
#     # delta_blc = delta
#     # delta_trc = delta
#     # if blc[0] == 0 or blc[1] == 0:
#     #     delta_blc = -1
#     # if trc[0] == array.shape[0] or trc[1] == array.shape[1]:
#     #     delta_trc = 0
#     blc = (int(blc[0] - delta), int(blc[1] - delta))
#     trc = (int(trc[0] + delta), int(trc[1] + delta))
#     return blc, trc


def check_bbox(blc, trc, image_size):
    """
    :note:
        This can make quadratic image rectangular.
    """
    # If some bottom corner coordinate become negative
    blc = list(blc)
    trc = list(trc)
    if blc[0] < 0:
        blc[0] = 0
    if blc[1] < 0:
        blc[1] = 0
    # If some top corner coordinate become large than image size
    if trc[0] > image_size:
        delta = abs(trc[0]-image_size)
        blc[0] -= delta
        # Check if shift have not made it negative
        if blc[0] < 0 and trc[0] > image_size:
            blc[0] = 0
        trc[0] -= delta
    if trc[1] > image_size:
        delta = abs(trc[1]-image_size)
        blc[1] -= delta
        # Check if shift have not made it negative
        if blc[1] < 0 and trc[1] > image_size:
            blc[1] = 0
        trc[1] -= delta
    return tuple(blc), tuple(trc)


def find_bbox(array, level, min_maxintensity_mjyperbeam=None, min_area_pix=None,
              delta=0.):
    """
    Find bounding box for part of image containing source.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_maxintensity_mjyperbeam:
        Minimum of the maximum intensity in the region to include.
    :param min_area_pix:
        Minimum area for region to include.
    :param delta: (optional)
        Extra space to add symmetrically [pixels]. (default: ``0``)
    :return:
        Tuples of BLC & TRC.

    :note:
        This is BLC, TRC for numpy array (i.e. transposed source map as it
        conventionally seen on VLBI maps).
    """
    if min_maxintensity_mjyperbeam is None:
        min_maxintensity_mjyperbeam = 3*level
    if min_area_pix is None:
        min_area_pix = 100
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    for prop in props:
        if prop.max_intensity > min_maxintensity_mjyperbeam/1000 and prop.area > min_area_pix:
            signal_props.append(prop)

    # Sometimes no regions are found. In that case return full image
    if not signal_props:
        return (0, 0,), (array.shape[1], array.shape[1],)

    blcs = list()
    trcs = list()

    for prop in signal_props:
        bbox = prop.bbox
        blc = (int(bbox[1]), int(bbox[0]))
        trc = (int(bbox[3]), int(bbox[2]))
        blcs.append(blc)
        trcs.append(trc)

    min_blc_0 = min([blc[0] for blc in blcs])
    min_blc_1 = min([blc[1] for blc in blcs])
    max_trc_0 = max([trc[0] for trc in trcs])
    max_trc_1 = max([trc[1] for trc in trcs])
    blc_rec = (min_blc_0-delta, min_blc_1-delta,)
    trc_rec = (max_trc_0+delta, max_trc_1+delta,)

    blc_rec_ = blc_rec
    trc_rec_ = trc_rec
    blc_rec_, trc_rec_ = check_bbox(blc_rec_, trc_rec_, array.shape[0])

    # Enlarge 10% each side
    delta_ra = abs(trc_rec[0]-blc_rec[0])
    delta_dec = abs(trc_rec[1]-blc_rec[1])
    blc_rec = (blc_rec[0] - int(0.1*delta_ra), blc_rec[1] - int(0.1*delta_dec))
    trc_rec = (trc_rec[0] + int(0.1*delta_ra), trc_rec[1] + int(0.1*delta_dec))

    blc_rec, trc_rec = check_bbox(blc_rec, trc_rec, array.shape[0])

    return (int(blc_rec[0]), int(blc_rec[1])), (int(trc_rec[0]), int(trc_rec[1]))


# FIXME: When plotting w & wo colors it flips axes! When only contours are
# plotted then it flips axis with negative increment!
# TODO: Implement plotting w/o coordinates - in pixels. Use pixel numbers as
# coordinates.
# TODO: Make possible use ``blc`` & ``trc`` in mas.
# TODO: Plot components from difmap-style txt-file or instances of ``Component``
# class.
# Beam BPA in deg
def plot(contours=None, colors=None, vectors=None, vectors_values=None, x=None,
         y=None, blc=None, trc=None, cmap='hsv', interp=None, abs_levels=None,
         rel_levels=None, min_abs_level=None, min_rel_level=None, k=2, vinc=2,
         show_beam=False, beam_place='ll', beam=None, contours_mask=None,
         colors_mask=None, log_color=False, log_offset=None, dynamic_range=1e+03,
         vectors_mask=None, plot_title=None, title_fontsize="medium", title_loc="left", color_clim=None,
         outfile=None, outdir=None, ext='png', close=False, slice_points=None,
         colorbar_label=None, show=True, contour_color='k',
         beam_edge_color='black', beam_face_color='green', beam_alpha=0.3,
         show_points=None, components=None, components_errors=None, slice_color='black',
         plot_colorbar=True, label_size=10, ra_range=None, dec_range=None,
         fig=None, axes=None, contour_linewidth=0.5, vector_color="black",
         n_discrete_colors=None, fixed_component_color="deepskyblue",
         show_xlabel_on_current_axes=False, show_ylabel_on_current_axes=False,
         vector_scale=None, components_facecolor=None, revert_xaxis=True):
    """
    Plot image(s).

    :param contours: (optional)
        Numpy 2D array (possibly masked) that should be plotted using contours.
    :param colors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using colors.
    :param vectors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using vectors.
    :param vectors_values: (optional)
        Numpy 2D array (possibly masked) that should be used as vector's lengths
        when plotting ``vectors`` array.
    :param x: (optional)
        Iterable of x-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param y: (optional)
        Iterable of y-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param blc: (optional)
        Iterable of two values for Bottom Left Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(1, 1)``. (default:
        ``None``)
    :param trc: (optional)
        Iterable of two values for Top Right Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(image_size,
        image_size)``. (default: ``None``)
    :param cmap: (optional)
        Colormap to use for plotting colors. Available color maps could be
        printed using ``sorted(m for m in plt.cm.datad if not
        m.endswith("_r"))`` where ``plt`` is imported ``matplotlib.pyplot``.
        For further details on plotting available colormaps see
        http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html.
        (default: ``hsv``)
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``factor``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``factor``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param k: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param show_beam: (optional)
        Convertable to boolean. Should we plot beam in corner? (default:
        ``False``)
    :param beam_corner: (optional)
        Place (corner) where to plot beam on map. One of ('ll', 'lr', 'ul',
        'ur') where first letter means lower/upper and second - left/right.
        (default: ``ll'')
    :param beam: (optional)
        If ``show_beam`` is True then ``beam`` should be iterable of major axis,
        minor axis [mas] and beam positional angle [deg]. If no coordinates are
        supplied then beam parameters must be in pixels.
    :param colorbar_label: (optional)
        String to label colorbar. If ``None`` then don't label. (default:
        ``None``)
    :param slice_points: (optional)
        Iterable of 2 coordinates (``y``, ``x``) [mas] to plot slice. If
        ``None`` then don't plot slice. (default: ``None``)
    :param show_points: (optional)
        Iterable of 2 coordinates (``y``, ``x``) [mas] to plot points. If
        ``None`` then don't plot points. (default: ``None``)
    :param plot_colorbar: (optional)
        If colors is set then should we plot colorbar? (default: ``True``).

    :note:
        ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
        converted to python-like zero-indexing. If none are specified then use
        default values. All images plotted must have the same shape.
    """
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    image = None
    if contours is not None:
        image = contours
    elif colors is not None and image is None:
        image = colors
    elif vectors is not None and image is None:
        image = vectors

    if image is None:
        raise Exception("No images to plot!")
    if x is None:
        x = np.arange(image.shape[0])
        factor_x = 1
    else:
        factor_x = 1. / mas_to_rad
    if y is None:
        y = np.arange(image.shape[1])
        factor_y = 1
    else:
        factor_y = 1. / mas_to_rad

    # Set BLC & TRC
    blc = blc or (1, 1,)
    trc = trc or image.shape

    # TODO: Some diff from calculon
    if blc[0] == 0:
        blc = (blc[0]+1, blc[1])
    if blc[1] == 0:
        blc = (blc[0], blc[1]+1)

    # Use ``-1`` because user expect AIPS-like behavior of ``blc`` & ``trc``
    x_slice = slice(blc[1] - 1, trc[1], None)
    y_slice = slice(blc[0] - 1, trc[0],  None)

    # Create coordinates
    imsize_x = x_slice.stop - x_slice.start
    imsize_y = y_slice.stop - y_slice.start
    # In mas (if ``x`` & ``y`` were supplied in rad) or in pixels (if no ``x`` &
    # ``y`` were supplied)
    x_ = x[x_slice] * factor_x
    y_ = y[y_slice] * factor_y
    # With this coordinates are plotted as in Zhenya's map
    # x_ *= -1.
    # y_ *= -1.
    # Coordinates for plotting
    x = np.linspace(x_[0], x_[-1], imsize_x)
    y = np.linspace(y_[0], y_[-1], imsize_y)

    plot_pixel_size = np.hypot((x[1]-x[0]), (y[1]-y[0]))

    # Optionally mask arrays
    if contours is not None and contours_mask is not None:
        contours = np.ma.array(contours, mask=contours_mask)
    if colors is not None and colors_mask is not None:
        colors = np.ma.array(colors, mask=colors_mask)
    if vectors is not None and vectors_mask is not None:
        vectors = np.ma.array(vectors, mask=vectors_mask)

    # Actually plotting
    if fig is None and axes is None:
        fig = plt.figure()
        fig.set_size_inches(4.5, 3.5)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel(r'Rel. RA (mas)')
        ax.set_ylabel(r'Rel. DEC (mas)')
    if axes is not None:
        ax = axes
        if show_xlabel_on_current_axes:
            ax.set_xlabel(r'Rel. RA (mas)')
        if show_ylabel_on_current_axes:
            ax.set_ylabel(r'Rel. DEC (mas)')
    else:
        ax = fig.get_axes()[0]
        ax.set_xlabel(r'Rel. RA (mas)')
        ax.set_ylabel(r'Rel. DEC (mas)')

    if ra_range:
        ax.set_xlim(ra_range)
    if dec_range:
        ax.set_ylim(dec_range)

    # Plot contours
    if contours is not None:
        # If no absolute levels are supplied then construct them
        if abs_levels is None:
            # print("constructing absolute levels for contours...")
            max_level = np.nanmax(contours[x_slice, y_slice])
            # from given relative levels
            if rel_levels is not None:
                # print("from relative levels...")
                # Build levels (``pyplot.contour`` takes only absolute values)
                abs_levels = [-max_level] + [max_level * i for i in rel_levels]
                # If given only min_abs_level & increment factor ``k``
            else:
                # from given minimal absolute level
                if min_abs_level is not None:
                    # print("from minimal absolute level...")
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                # from given minimal relative level
                elif min_rel_level is not None:
                    # print("from minimal relative level...")
                    min_abs_level = min_rel_level * max_level / 100.
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                abs_levels = [-min_abs_level] + [min_abs_level * k ** i for i in
                                                 range(n_max)]
            print("Constructed absolute levels are: {}".format(abs_levels))

        if fig is None and axes is None:
            extent = [y[0], y[-1], x[-1], x[0]]
        else:
            extent = [y[0], y[-1], x[0], x[-1]]
        co = ax.contour(y, x, contours[x_slice, y_slice], abs_levels,
                        colors=contour_color, extent=extent,
                        linewidths=contour_linewidth)
        if revert_xaxis:
            print("Inverting xaxis!")
            ax.invert_xaxis()
        # Make colorbar for contours if no colors is supplied
        if colors is None:
            if plot_colorbar:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="10%", pad=0.00)
                cb = fig.colorbar(co, cax=cax)
                cb.set_label(colorbar_label)

    if colors is not None:
        if n_discrete_colors is not None:
            cmap = cm.get_cmap(cmap, int(n_discrete_colors))
        if interp is None:
            interp = "none"

        if log_color:
            if (colors < 0.0).any():
                print('clipping values less than 0 in display')
                colors[colors < 0.0] = 0.0
            if log_offset is not None:
                colors = np.log10(colors + log_offset/dynamic_range)
            else:
                colors = np.log10(colors + np.max(colors)/dynamic_range)

        im = ax.imshow(colors[x_slice, y_slice], interpolation=interp,
                       origin='lower', extent=[y[0], y[-1], x[0], x[-1]],
                       cmap=plt.get_cmap(cmap), clim=color_clim)

    if vectors is not None:
        if vectors_values is not None:
            # TODO: Does "-" sign because of RA increases to the left actually?
            # VLBIers do count angles from North to negative RA.
            u = -vectors_values[x_slice, y_slice] * np.sin(vectors[x_slice,
                                                                   y_slice])
            v = vectors_values[x_slice, y_slice] * np.cos(vectors[x_slice,
                                                                  y_slice])
        else:
            u = -np.sin(vectors[x_slice, y_slice])
            v = np.cos(vectors[x_slice, y_slice])

        if vectors_mask is not None:
            u = np.ma.array(u, mask=vectors_mask[x_slice, y_slice])
            v = np.ma.array(v, mask=vectors_mask[x_slice, y_slice])
        vec = ax.quiver(y[::vinc], x[::vinc], u[::vinc, ::vinc],
                        v[::vinc, ::vinc], angles='uv',
                        units='xy', headwidth=0., headlength=0., scale=vector_scale,
                        width=0.015, headaxislength=0., color=vector_color)
    # Set equal aspect
    ax.set_aspect('equal')

    if slice_points is not None:
        for single_slice in slice_points:
            ax.plot([single_slice[0][0], single_slice[1][0]],
                    [single_slice[0][1], single_slice[1][1]], color=slice_color)

    if show_points is not None:
        for point in show_points:
            ax.plot(point[0], point[1], '.k')

    if plot_title:
        title = ax.set_title(plot_title, fontsize=title_fontsize, loc=title_loc)
    # Add colorbar if plotting colors
    if colors is not None:
        if plot_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.00)
            if fig is None and axes is not None:
                fig = ax.get_figure()
            cb = fig.colorbar(im, cax=cax)
            if colorbar_label is not None:
                cb.set_label(colorbar_label)

    if show_beam:

        # https://matplotlib.org/stable/tutorials/text/annotations.html
        # from matplotlib.patches import Ellipse
        # from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
        #
        # fig, ax = plt.subplots()
        # box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
        # el = Ellipse((0, 0), width=0.1, height=0.4, angle=30)  # in data coordinates!
        # box.drawing_area.add_artist(el)
        # ax.add_artist(box)


        e_height = max(beam[0], beam[1])
        e_width = min(beam[0], beam[1])
        r_min = e_height / 2
        r_min = e_height
        if beam_place == 'lr':
            if y[0]-y[1] > 0:
                y_c = y[-1] + r_min
            else:
                y_c = y[-1] - r_min
            if x[0]-x[1] > 0:
                x_c = x[0] - r_min
            else:
                x_c = x[0] + r_min
        elif beam_place == 'll':
            if y[0]-y[1] > 0:
                y_c = y[0] - r_min
            else:
                y_c = y[0] + r_min
            if x[0]-x[1] > 0:
                x_c = x[0] - r_min
            else:
                x_c = x[0] + r_min
        elif beam_place == 'ul':
            if y[0]-y[1] > 0:
                y_c = y[0] - r_min
            else:
                y_c = y[0] + r_min
            if x[0]-x[1] > 0:
                x_c = x[-1] + r_min
            else:
                x_c = x[-1] - r_min
        elif beam_place == 'ur':
            if y[0]-y[1] > 0:
                y_c = y[-1] + r_min
            else:
                y_c = y[-1] - r_min
            if x[0]-x[1] > 0:
                x_c = x[-1] + r_min
            else:
                x_c = x[-1] - r_min
        else:
            raise Exception

        # FIXME: check how ``bpa`` should be plotted
        # Here width & height - total widths
        e = Ellipse((y_c, x_c), e_width, e_height, angle=-beam[2],
                    edgecolor=beam_edge_color, facecolor=beam_face_color,
                    alpha=beam_alpha)
        print("Plotting BEAM BPA = {:.2f} deg".format(beam[2]))
        ax.add_patch(e)

    if components:
        if components_facecolor is None:
            facecolor = "red"
        else:
            facecolor = components_facecolor
        for comp in components:
            if np.any(comp._fixed):
                is_some_parameter_fixed = True
            else:
                is_some_parameter_fixed = False

            # RA
            x_c = -comp.p_all[1]
            # DEC
            y_c = -comp.p_all[2]
            # len used for all parameters (even fixed)
            if len(comp) == 6:
                e_height = comp.p_all[3]
                e_width = comp.p_all[3] * comp.p_all[4]
                if e_height < 0.25*plot_pixel_size and e_width < 0.25*plot_pixel_size:
                    if components_facecolor is None:
                        facecolor = "green"
                if e_height < 0.25*plot_pixel_size:
                    e_height = 0.25*plot_pixel_size
                if e_width < 0.25*plot_pixel_size:
                    e_width = 0.25*plot_pixel_size*comp.p_all[4]
                else:
                    if components_facecolor is None:
                        facecolor = "red"
                if comp.p_all[0] < 0:
                    if components_facecolor is None:
                        facecolor = "blue"

                if is_some_parameter_fixed:
                    if components_facecolor is None:
                        facecolor = fixed_component_color

                # FIXME: Here must be total length of vertical/horizontal axis
                e = Ellipse((x_c, y_c), e_width, e_height,
                            angle=90.0-180*comp.p_all[5]/np.pi,
                            edgecolor=beam_edge_color, facecolor=facecolor,
                            alpha=beam_alpha)
            elif len(comp) == 4:
                # It is radius so dividing in 2
                c_size = comp.p_all[3]/2.0
                if c_size < 0.25*plot_pixel_size:
                    c_size = 0.25*plot_pixel_size
                    if components_facecolor is None:
                        facecolor = "green"
                else:
                    if components_facecolor is None:
                        facecolor = "red"
                if comp.p_all[0] < 0:
                    if components_facecolor is None:
                        facecolor = "blue"

                if is_some_parameter_fixed:
                    if components_facecolor is None:
                        facecolor = fixed_component_color

                e = Circle((x_c, y_c), c_size,
                            edgecolor=beam_edge_color, facecolor=facecolor,
                            alpha=beam_alpha)
            elif len(comp) == 3:

                if is_some_parameter_fixed:
                    facecolor = fixed_component_color

                e = Circle((x_c, y_c), plot_pixel_size,
                            edgecolor=beam_edge_color, facecolor='green',
                            alpha=beam_alpha)
            else:
                raise Exception("Only Point, Circle or Ellipse components are"
                                " plotted")
            ax.add_patch(e)

    if components_errors is not None:
        facecolor = "gray"
        # Each components represent error ellipse
        for comp in components_errors:
            # RA
            x_c = -comp.p_all[1]
            # DEC
            y_c = -comp.p_all[2]
            e_height = comp.p_all[3]
            e_width = comp.p_all[3] * comp.p_all[4]
            e = Ellipse((x_c, y_c), e_width, e_height,
                        angle=90.0-180*comp.p_all[5]/np.pi,
                        edgecolor=beam_edge_color, facecolor=facecolor,
                        alpha=beam_alpha)
            ax.add_patch(e)

    # Saving output
    if outfile:
        if outdir is None:
            outdir = '.'
        # If the directory does not exist, create it
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = os.path.join(outdir, outfile)
        print("Saving to {}.{}".format(path, ext))
        plt.savefig("{}.{}".format(path, ext), bbox_inches='tight', dpi=500)

    if show and fig is not None:
        fig.show()
    if show and fig is None:
        plt.show()
    if close:
        plt.close()

    if fig is not None:
        return fig



def plot2(contours=None, colors=None, vectors=None, vectors_values=None, x=None,
         y=None, blc=None, trc=None, cmap='hsv', abs_levels=None,
         rel_levels=None, min_abs_level=None, min_rel_level=None, k=2, vinc=2,
         show_beam=False, beam_place='ll', beam=None, contours_mask=None,
         colors_mask=None, vectors_mask=None, plot_title=None, color_clim=None,
         outfile=None, outdir=None, ext='png', close=False, slice_points=None,
         colorbar_label=None, show=True, contour_color='k',
         beam_edge_color='black', beam_face_color='green', beam_alpha=0.3,
         show_points=None, components=None, components_errors=None, slice_color='black',
         plot_colorbar=True, label_size=12, ra_range=None, dec_range=None,
         fig=None, axes=None, contour_linewidth=0.5, vector_color="black",
         n_discrete_colors=None, fixed_component_color="deepskyblue"):
    """
    Plot image(s).

    :param contours: (optional)
        Numpy 2D array (possibly masked) that should be plotted using contours.
    :param colors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using colors.
    :param vectors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using vectors.
    :param vectors_values: (optional)
        Numpy 2D array (possibly masked) that should be used as vector's lengths
        when plotting ``vectors`` array.
    :param x: (optional)
        Iterable of x-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param y: (optional)
        Iterable of y-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param blc: (optional)
        Iterable of two values for Bottom Left Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(1, 1)``. (default:
        ``None``)
    :param trc: (optional)
        Iterable of two values for Top Right Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(image_size,
        image_size)``. (default: ``None``)
    :param cmap: (optional)
        Colormap to use for plotting colors. Available color maps could be
        printed using ``sorted(m for m in plt.cm.datad if not
        m.endswith("_r"))`` where ``plt`` is imported ``matplotlib.pyplot``.
        For further details on plotting available colormaps see
        http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html.
        (default: ``hsv``)
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``factor``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``factor``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param k: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param show_beam: (optional)
        Convertable to boolean. Should we plot beam in corner? (default:
        ``False``)
    :param beam_corner: (optional)
        Place (corner) where to plot beam on map. One of ('ll', 'lr', 'ul',
        'ur') where first letter means lower/upper and second - left/right.
        (default: ``ll'')
    :param beam: (optional)
        If ``show_beam`` is True then ``beam`` should be iterable of major axis,
        minor axis [mas] and beam positional angle [deg]. If no coordinates are
        supplied then beam parameters must be in pixels.
    :param colorbar_label: (optional)
        String to label colorbar. If ``None`` then don't label. (default:
        ``None``)
    :param slice_points: (optional)
        Iterable of 2 coordinates (``y``, ``x``) [mas] to plot slice. If
        ``None`` then don't plot slice. (default: ``None``)
    :param show_points: (optional)
        Iterable of 2 coordinates (``y``, ``x``) [mas] to plot points. If
        ``None`` then don't plot points. (default: ``None``)
    :param plot_colorbar: (optional)
        If colors is set then should we plot colorbar? (default: ``True``).

    :note:
        ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
        converted to python-like zero-indexing. If none are specified then use
        default values. All images plotted must have the same shape.
    """
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    image = None
    if contours is not None:
        image = contours
    elif colors is not None and image is None:
        image = colors
    elif vectors is not None and image is None:
        image = vectors

    if image is None:
        raise Exception("No images to plot!")
    if x is None:
        x = np.arange(image.shape[0])
        factor_x = 1
    else:
        factor_x = 1. / mas_to_rad
    if y is None:
        y = np.arange(image.shape[1])
        factor_y = 1
    else:
        factor_y = 1. / mas_to_rad

    # Set BLC & TRC
    blc = blc or (1, 1,)
    trc = trc or image.shape

    # TODO: Some diff from calculon
    if blc[0] == 0:
        blc = (blc[0]+1, blc[1])
    if blc[1] == 0:
        blc = (blc[0], blc[1]+1)

    # Use ``-1`` because user expect AIPS-like behavior of ``blc`` & ``trc``
    x_slice = slice(blc[1] - 1, trc[1], None)
    y_slice = slice(blc[0] - 1, trc[0],  None)

    # Create coordinates
    imsize_x = x_slice.stop - x_slice.start
    imsize_y = y_slice.stop - y_slice.start
    # In mas (if ``x`` & ``y`` were supplied in rad) or in pixels (if no ``x`` &
    # ``y`` were supplied)
    x_ = x[x_slice] * factor_x
    y_ = y[y_slice] * factor_y
    # With this coordinates are plotted as in Zhenya's map
    # x_ *= -1.
    # y_ *= -1.
    # Coordinates for plotting
    x = np.linspace(x_[0], x_[-1], imsize_x)
    y = np.linspace(y_[0], y_[-1], imsize_y)

    plot_pixel_size = np.hypot((x[1]-x[0]), (y[1]-y[0]))

    # Optionally mask arrays
    if contours is not None and contours_mask is not None:
        contours = np.ma.array(contours, mask=contours_mask)
    if colors is not None and colors_mask is not None:
        colors = np.ma.array(colors, mask=colors_mask)
    if vectors is not None and vectors_mask is not None:
        vectors = np.ma.array(vectors, mask=vectors_mask)

    # Actually plotting
    if fig is None and axes is None:
        fig = plt.figure()
        fig.set_size_inches(4.5, 3.5)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel(r'Relative R.A. (mas)')
        ax.set_ylabel(r'Relative Decl. (mas)')
    if axes is not None:
        ax = axes
    else:
        ax = fig.get_axes()[0]
        ax.set_xlabel(r'Relative R.A. (mas)')
        ax.set_ylabel(r'Relative Decl. (mas)')

    if ra_range:
        ax.set_xlim(ra_range)
    if dec_range:
        ax.set_ylim(dec_range)

    # Plot contours
    if contours is not None:
        # If no absolute levels are supplied then construct them
        if abs_levels is None:
            # print("constructing absolute levels for contours...")
            max_level = contours[x_slice, y_slice].max()
            # from given relative levels
            if rel_levels is not None:
                # print("from relative levels...")
                # Build levels (``pyplot.contour`` takes only absolute values)
                abs_levels = [-max_level] + [max_level * i for i in rel_levels]
                # If given only min_abs_level & increment factor ``k``
            else:
                # from given minimal absolute level
                if min_abs_level is not None:
                    # print("from minimal absolute level...")
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                # from given minimal relative level
                elif min_rel_level is not None:
                    # print("from minimal relative level...")
                    min_abs_level = min_rel_level * max_level / 100.
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                abs_levels = [-min_abs_level] + [min_abs_level * k ** i for i in
                                                 range(n_max)]
            print("Constructed absolute levels are: {}".format(abs_levels))

        if fig is None and axes is None:
            extent = [y[0], y[-1], x[-1], x[0]]
        else:
            extent = [y[0], y[-1], x[0], x[-1]]
        co = ax.contour(y, x, contours[x_slice, y_slice], abs_levels,
                        colors=contour_color, extent=extent,
                        linewidths=contour_linewidth)
        # if fig is None:
        ax.invert_xaxis()
        # Make colorbar for contours if no colors is supplied
        if colors is None:
            if plot_colorbar:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="10%", pad=0.00)
                cb = fig.colorbar(co, cax=cax)
                cb.set_label(colorbar_label)
    if colors is not None:
        if n_discrete_colors is not None:
            cmap = cm.get_cmap(cmap, int(n_discrete_colors))
        im = ax.imshow(colors[x_slice, y_slice], interpolation='none',
                       origin='lower', extent=[y[0], y[-1], x[0], x[-1]],
                       cmap=plt.get_cmap(cmap), clim=color_clim)
    if vectors is not None:
        if vectors_values is not None:
            # TODO: Does "-" sign because of RA increases to the left actually?
            # VLBIers do count angles from North to negative RA.
            u = -vectors_values[x_slice, y_slice] * np.sin(vectors[x_slice,
                                                                   y_slice])
            v = vectors_values[x_slice, y_slice] * np.cos(vectors[x_slice,
                                                                  y_slice])
        else:
            u = -np.sin(vectors[x_slice, y_slice])
            v = np.cos(vectors[x_slice, y_slice])

        if vectors_mask is not None:
            u = np.ma.array(u, mask=vectors_mask[x_slice, y_slice])
            v = np.ma.array(v, mask=vectors_mask[x_slice, y_slice])
        vec = ax.quiver(y[::vinc], x[::vinc], u[::vinc, ::vinc],
                        v[::vinc, ::vinc], angles='uv',
                        units='xy', headwidth=0., headlength=0., scale=None,
                        width=0.05, headaxislength=0., color=vector_color)
    # Set equal aspect
    ax.set_aspect('equal')

    if slice_points is not None:
        for single_slice in slice_points:
            ax.plot([single_slice[0][0], single_slice[1][0]],
                    [single_slice[0][1], single_slice[1][1]], color=slice_color)

    if show_points is not None:
        for point in show_points:
            ax.plot(point[0], point[1], '.k')

    if plot_title:
        title = ax.set_title(plot_title, fontsize='large')
    # Add colorbar if plotting colors
    if colors is not None:
        if plot_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.00)
            cb = fig.colorbar(im, cax=cax)
            if colorbar_label is not None:
                cb.set_label(colorbar_label)

    if show_beam:
        from matplotlib.patches import Ellipse
        e_height = beam[0]
        e_width = beam[1]
        r_min = e_height / 2
        if beam_place == 'lr':
            if y[0]-y[1] > 0:
                y_c = y[-1] + r_min
            else:
                y_c = y[-1] - r_min
            if x[0]-x[1] > 0:
                x_c = x[0] - r_min
            else:
                x_c = x[0] + r_min
        elif beam_place == 'll':
            if y[0]-y[1] > 0:
                y_c = y[0] - r_min
            else:
                y_c = y[0] + r_min
            if x[0]-x[1] > 0:
                x_c = x[0] - r_min
            else:
                x_c = x[0] + r_min
        elif beam_place == 'ul':
            if y[0]-y[1] > 0:
                y_c = y[0] - r_min
            else:
                y_c = y[0] + r_min
            if x[0]-x[1] > 0:
                x_c = x[-1] + r_min
            else:
                x_c = x[-1] - r_min
        elif beam_place == 'ur':
            if y[0]-y[1] > 0:
                y_c = y[-1] + r_min
            else:
                y_c = y[-1] - r_min
            if x[0]-x[1] > 0:
                x_c = x[-1] + r_min
            else:
                x_c = x[-1] - r_min
        else:
            raise Exception

        # FIXME: check how ``bpa`` should be plotted
        e = Ellipse((y_c, x_c), e_width, e_height, angle=-beam[2],
                    edgecolor=beam_edge_color, facecolor=beam_face_color,
                    alpha=beam_alpha)
        ax.add_patch(e)

    if components:
        facecolor = "red"
        for comp in components:
            if np.any(comp._fixed):
                is_some_parameter_fixed = True
            else:
                is_some_parameter_fixed = False

            # RA
            x_c = -comp.p_all[1]
            # DEC
            y_c = -comp.p_all[2]
            # len used for all parameters (even fixed)
            if len(comp) == 6:
                e_height = comp.p_all[3]
                e_width = comp.p_all[3] * comp.p_all[4]
                if e_height < 0.25*plot_pixel_size and e_width < 0.25*plot_pixel_size:
                    facecolor = "green"
                if e_height < 0.25*plot_pixel_size:
                    e_height = 0.25*plot_pixel_size
                if e_width < 0.25*plot_pixel_size:
                    e_width = 0.25*plot_pixel_size*comp.p_all[4]
                else:
                    facecolor = "red"
                if comp.p_all[0] < 0:
                    facecolor = "blue"

                if is_some_parameter_fixed:
                    facecolor = fixed_component_color

                # FIXME: Here must be total length of vertical/horizontal axis
                e = Ellipse((x_c, y_c), e_width, e_height,
                            angle=90.0-180*comp.p_all[5]/np.pi,
                            edgecolor=beam_edge_color, facecolor=facecolor,
                            alpha=beam_alpha)
            elif len(comp) == 4:
                # It is radius so dividing in 2
                c_size = comp.p_all[3]/2.0
                if c_size < 0.25*plot_pixel_size:
                    c_size = 0.25*plot_pixel_size
                    facecolor = "green"
                else:
                    facecolor = "red"
                if comp.p_all[0] < 0:
                    facecolor = "blue"

                if is_some_parameter_fixed:
                    facecolor = fixed_component_color

                e = Circle((x_c, y_c), c_size,
                            edgecolor=beam_edge_color, facecolor=facecolor,
                            alpha=beam_alpha)
            elif len(comp) == 3:

                if is_some_parameter_fixed:
                    facecolor = fixed_component_color

                e = Circle((x_c, y_c), plot_pixel_size,
                            edgecolor=beam_edge_color, facecolor='green',
                            alpha=beam_alpha)
            else:
                raise Exception("Only Point, Circle or Ellipse components are"
                                " plotted")
            ax.add_patch(e)

    if components_errors is not None:
        facecolor = "gray"
        # Each components represent error ellipse
        for comp in components_errors:
            # RA
            x_c = -comp.p_all[1]
            # DEC
            y_c = -comp.p_all[2]
            e_height = comp.p_all[3]
            e_width = comp.p_all[3] * comp.p_all[4]
            e = Ellipse((x_c, y_c), e_width, e_height,
                        angle=90.0-180*comp.p_all[5]/np.pi,
                        edgecolor=beam_edge_color, facecolor=facecolor,
                        alpha=beam_alpha)
            ax.add_patch(e)

    # Saving output
    if outfile:
        if outdir is None:
            outdir = '.'
        # If the directory does not exist, create it
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = os.path.join(outdir, outfile)
        print("Saving to {}.{}".format(path, ext))
        plt.savefig("{}.{}".format(path, ext), bbox_inches='tight', dpi=500)

    if show and fig is not None:
        fig.show()
    if close:
        plt.close()

    if fig is not None:
        return fig




class BasicImage(object):
    """
    Image class that implements basic image functionality that physical scale
    free.
    """
    def __init__(self):
        self.imsize = None
        self._image = None

    def _construct(self, **kwargs):
        try:
            self.imsize = kwargs["imsize"]
        except KeyError:
            raise Exception
        self._image = np.zeros(self.imsize, dtype=float)

    def from_hdulist(self, hdulist):
        image_params = get_fits_image_info_from_hdulist(hdulist)
        self._construct(**image_params)
        pr_hdu = get_hdu_from_hdulist(hdulist)
        self.image = pr_hdu.data.squeeze()

    def from_fits(self, fname):
        hdulist = pf.open(fname)
        self.from_hdulist(hdulist)

    @property
    def image(self):
        """
        Shorthand for image array.
        """
        return self._image

    @image.setter
    def image(self, image):
        if isinstance(image, Image):
            if self == image:
                self._image = image.image.copy()
            else:
                raise Exception("Images have incompatible parameters!")
        # If ``image`` is array-like
        else:
            image = np.atleast_2d(image).copy()
            if not self.imsize == np.shape(image):
                raise Exception("Images have incompatible parameters!")
            self._image = image

    def __eq__(self, other):
        """
        Compares current instance of ``BasicImage`` class with other instance.
        """
        return self.imsize == other.imsize

    def __ne__(self, image):
        """
        Compares current instance of ``BasicImage`` class with other instance.
        """
        return self.imsize != image.imsize

    def __add__(self, image):
        """
        Sums current instance of ``BasicImage`` class with other instance.
        """
        if self == image:
            self.image += image.image
        else:
            raise Exception("Different image parameters")
        return self

    def __mul__(self, other):
        """
        Multiply current instance of ``BasicImage`` class with other instance or
        some number.
        """
        if isinstance(other, BasicImage):
            if self == other:
                self.image *= other.image
            else:
                raise Exception("Different image parameters")
        else:
            self.image *= other
        return self

    def __sub__(self, other):
        """
        Substruct from current instance of ``BasicImage`` class other instance
        or some number.
        """
        if isinstance(other, BasicImage):
            if self == other:
                self._image -= other.image
            else:
                raise Exception("Different image parameters")
        else:
            self._image -= other
        return self

    def __div__(self, other):
        """
        Divide current instance of ``BasicImage`` class on other instance or
        some number.
        """
        if isinstance(other, BasicImage):
            if self == other:
                self.image /= other.image
            else:
                raise Exception("Different image parameters")
        else:
            self.image /= other
        return self

    # TODO: In subclasses pixels got sizes so one can use physical sizes as
    # ``region`` parameters. Subclasses should extend this method.
    def rms(self, region=None, do_plot=False, **hist_kwargs):
        """
        Method that calculate rms for image region.

        :param region: (optional)
            Region to include in rms calculation. Or (blc[0], blc[1], trc[0],
            trc[1],) or (center[0], center[1], r, None,). If ``None`` then use
            all image in rms calculation. Default ``None``.
        :param do_plot: (optional)
            Plot histogram of image values? (default: ``False``)
        :param hist_kwargs: (optional)
            Any keyword arguments that get passed to ``plt.hist``.
        :return:
            rms value.
        """
        mask = np.zeros(self.image.shape, dtype=bool)
        if region is not None:
            mask = create_mask(self.image.shape, region)
        masked_image = np.ma.array(self.image, mask=~mask)

        if do_plot:
            plt.hist(masked_image.compressed(), **hist_kwargs)

        return np.ma.std(masked_image.ravel())

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
        return signal.fftconvolve(self._image, to_convolve, mode='same')

    # TODO: Implement Rayleigh (Rice) distributed noise for stokes I
    # FIXME: This is uncorrelated noise - that is too simple model
    def add_noise(self, std, df=None):
        size = self.imsize[0] * self.imsize[1]
        if df is None:
            rvs = np.random.normal(loc=0., scale=std, size=size)
        else:
            raise NotImplementedError
        rvs = rvs.reshape(self.imsize)
        self._image += rvs

    # TODO: Should i compare images before?
    # TODO: Implement several regions to include for each image
    # TODO: Implement masking clean components with ``mask_cc`` parameter
    def cross_correlate(self, image, region1=None, region2=None,
                        upsample_factor=100, extended_output=False,
                        mask_cc=False):
        """
        Cross-correlates current instance of ``Image`` with another instance
        using phase correlation.

        :param image:
            Instance of image class.
        :param region1: (optional)
            Region to EXCLUDE in current instance of ``Image``.
            Or (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
            None,) or (center[0], center[1], bmaj, e, bpa). Default ``None``.
        :param region2: (optional)
            Region to EXCLUDE in other instance of ``Image``. Or (blc[0],
            blc[1], trc[0], trc[1],) or (center[0], center[1], r, None,) or
            (center[0], center[1], bmaj, e, bpa). Default ``None``.
        :param upsample_factor: (optional)
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel. If ``1`` then no upsampling.
            (default: ``100``)
        :param extended_output: (optioinal)
            Output all information from ``register_translation``? (default:
            ``False``)
        :param mask_cc: (optional)
            If some of images is instance of ``CleanImage`` class - should we
            mask clean components instead of image array? (default: ``False``)

        :return:
            Array of shifts (subpixeled) in each direction or full information
            from ``register_translation`` depending on ``extended_output``.
        """
        image1 = self.image.copy()
        if region1 is not None:
            mask1 = create_mask(self.image.shape, region1)
            if mask_cc and isinstance(self, CleanImage):
                raise NotImplementedError()
            image1[mask1] = 0.
        image2 = image.image.copy()
        if region2 is not None:
            mask2 = create_mask(image.image.shape, region2)
            if mask_cc and isinstance(image, CleanImage):
                raise NotImplementedError()
            image2[mask2] = 0.
        # Cross-correlate images
        shift, error, diffphase = phase_cross_correlation(image1, image2,
                                                       upsample_factor=upsample_factor)
        result = shift
        if extended_output:
            result = (shift, error, diffphase)
        return result

    # TODO: Implement physical sizes as vertexes of slice in ``Image``
    def slice(self, pix1, pix2):
        """
        Method that returns slice of image along line.

        :param pix1:
            Iterable of coordinates of first pixel.
        :param pix2:
            Iterable of coordinates of second pixel.
        :return:
            Numpy array of image values for given slice.
        """
        length = int(round(np.hypot(pix2[0] - pix1[0], pix2[1] - pix1[1])))
        if pix2[0] < pix1[0]:
            x = np.linspace(pix2[0], pix1[0], length)[::-1]
        else:
            x = np.linspace(pix1[0], pix2[0], length)
        if pix2[1] < pix1[1]:
            y = np.linspace(pix2[1], pix1[1], length)[::-1]
        else:
            y = np.linspace(pix1[1], pix2[1], length)

        return self.image[v_round(x).astype(np.int), v_round(y).astype(np.int)]

    def _in_img(self, point, fixed_point, slope):
        x1, y1 = point
        y0, x0 = fixed_point
        if x1 > self.imsize[0]-1:
            x1 = self.imsize[0]-1
            y1 = y0 + slope*(x1-x0)
        if x1 < 0:
            x1 = 0
            y1 = y0 + slope*(x1-x0)
        if y1 > self.imsize[1]-1:
            y1 = self.imsize[1]-1
            x1 = x0 + 1/slope*(y1-y0)
        if y1 < 0:
            y1 = 0
            x1 = x0 + 1/slope*(y1-y0)
        return (int(round(x1)), int(round(y1)))
        


# TODO: Option for saving ``Image`` instance
# TODO: Default value of pixref - center of image.
class Image(BasicImage):
    """
    Class that represents images.
    """
    def __init__(self):
        super(Image, self).__init__()
        self.pixsize = None
        self.pixref = None
        self.pixrefval = None
        self.freq = None
        self.stokes = None

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        image = Image()
        image._construct(self.pixsize, self.pixref, self.stokes, self.freq,
                         self.pixrefval, imsize=self.imsize)
        image.image = self.image
        return image

    def _construct(self, pixsize, pixref, stokes, freq, pixrefval, **kwargs):
        super(Image, self)._construct(**kwargs)
        self.pixsize = pixsize
        try:
            self.pixref = pixref
        except KeyError:
            self.pixref = (int(self.imsize / 2), int(self.imsize / 2))
        self.stokes = stokes
        self.freq = freq
        self.dy, self.dx = self.pixsize
        self.y_c, self.x_c = self.pixref
        try:
            self.pixrefval = pixrefval
        except KeyError:
            self.pixrefval = (0., 0.)
        self.x_c_val, self.y_c_val = self.pixrefval
        # Create coordinate arrays
        xsize, ysize = self.imsize
        # x = np.linspace(0, xsize - 1, xsize)
        # There should be (1, ...)?
        x = np.linspace(0, xsize, xsize)
        # y = np.linspace(0, ysize - 1, ysize)
        y = np.linspace(0, ysize, ysize)
        xv, yv = np.meshgrid(x, y)
        x -= self.x_c
        xv -= self.x_c
        y -= self.y_c
        yv -= self.y_c
        x *= self.dx
        xv *= self.dx
        y *= self.dy
        yv *= self.dy
        self.x = x
        self.xv = xv
        self.y = y
        self.yv = yv

    def _convert_coordinate(self, point_mas):
        """
        Convert coordinates from image scale [mas] to pixels.

        :param point_mas:
            Iterable of coordinates of pixel [mas].
        :return:
            Tuples of coordinates for pixel [pixels].
        """
        ycoords = self.y / mas_to_rad
        xcoords = self.x / mas_to_rad

        y0 = np.argmin(np.abs(ycoords - point_mas[0]))
        x0 = np.argmin(np.abs(xcoords - point_mas[1]))
        return x0, y0

    def _convert_array_coordinate(self, coord):
        try:
            dec = (self.x/mas_to_rad)[coord[1]]
        except IndexError:
            dec = (self.x/mas_to_rad)[coord[1]-1]
        try:
            ra = (self.y/mas_to_rad)[coord[0]]
        except IndexError:
            ra = (self.y/mas_to_rad)[coord[0]-1]
        return dec, ra

    def _convert_array_bbox(self, blc, trc):
        dec0, ra0 = self._convert_array_coordinate(blc)
        dec1, ra1 = self._convert_array_coordinate(trc)
        min_dec = min(dec0, dec1)
        max_dec = max(dec0, dec1)
        min_ra = min(ra0, ra1)
        max_ra = max(ra0, ra1)
        return (min_dec, max_dec), (min_ra, max_ra)

    def _inv_convert_coordinate(self, point_pix):
        """
        Convert coordinates from pixels to image scale [mas].

        :param point_pix:
            Iterable of coordinates in pixel.
        :return:
            Tuples of image coordinates for pixel [mas].
        """
        raise NotImplementedError

    def _convert_coordinates(self, point1_mas, point2_mas):
        """
        Convert coordinates from image scale [mas] to pixels.

        :param point1_mas:
            Iterable of coordinates of first pixel [mas].
        :param point2_mas:
            Iterable of coordinates of second pixel [mas].
        :return:
            Tuples of coordinates for first & second pixel [pixels].
        """
        ycoords = self.y / mas_to_rad
        xcoords = self.x / mas_to_rad

        y0 = np.argmin(np.abs(ycoords - point1_mas[0]))
        y1 = np.argmin(np.abs(ycoords - point2_mas[0]))
        x0 = np.argmin(np.abs(xcoords - point1_mas[1]))
        x1 = np.argmin(np.abs(xcoords - point2_mas[1]))
        return (x0, y0), (x1, y1)

    def _inv_convert_coordinates(self, point1_pix, point2_pix):
        """
        Convert coordinates from pixels to image scale [mas].

        :param point1_pix:
            Iterable of coordinates of first pixel.
        :param point2_pix:
            Iterable of coordinates of second pixel.
        :return:
            Tuples of image coordinates for first & second pixel [mas].
        """
        raise NotImplementedError

    def slice(self, pix1=None, pix2=None, point1=None, point2=None):
        """
        Return slice of image in pixels or image coordinates.
        :param pix1: (optional)
            Iterable of coordinates of first pixel [pixels].
        :param pix2: (optional)
            Iterable of coordinates of second pixel [pixels].
        :param point1: (optional)
            Iterable of coordinates of first pixel [mas].
        :param point2: (optional)
            Iterable of coordinates of second pixel [mas].
        :return:
            Numpy array of image values for given slice.
        """
        if pix1 is not None and pix2 is not None:
            return super(Image, self).slice(pix1, pix2)
        if point1 is not None and point2 is not None:
            # Convert coordinates [mas] to pixels
            pix1, pix2 = self._convert_coordinates(point1, point2)
            return super(Image, self).slice(pix1, pix2)

    def __eq__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (super(Image, self).__eq__(other) and
                self.pixsize == other.pixsize)

    def __ne__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return super(Image, self).__ne__(other) or self.pixsize != other.pixsize

    @property
    def phys_size(self):
        """
        Shortcut for physical size of image.
        """
        return (self.imsize[0] * abs(self.pixsize[0]), self.imsize[1] *
                abs(self.pixsize[1]))

    # This method has no sense in ``BasicImage`` class as there are no physical
    # sizes here.
    def add_component(self, component, beam=None):
        component.add_to_image(self, beam=beam)

    def substract_component(self, component, beam=None):
        component.substract_from_image(self, beam=beam)

    def add_model(self, model, beam=None):
        if self.stokes != model.stokes:
            raise Exception
        model.add_to_image(self, beam=beam)

    def substract_model(self, model, beam=None):
        if self.stokes != model.stokes:
            raise Exception
        model.substract_from_image(self, beam=beam)

    def plot(self, blc=None, trc=None, clim=None, cmap=None, abs_levels=None,
             rel_levels=None, min_abs_level=None, min_rel_level=None, factor=2.,
             plot_color=False):
        """
        Plot image.

        :note:
            ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
            converted to python-like zero-indexing.

        """
        pass
        # plot(self.image, x=self.x, y=self.y, blc=blc, trc=trc, clim=clim,
        #      cmap=cmap, abs_levels=abs_levels, rel_levels=rel_levels,
        #      min_abs_level=min_abs_level, min_rel_level=min_rel_level,
        #      factor=factor, plot_color=plot_color)


# TODO: ``cc`` attribute should be collection of ``Model`` instances!
# TODO: Add method ``shift`` that shifts image (CCs and residulas). Is it better
# to shift in uv-domain?
# FIXME: Why this bitch does not inherit __div__ from ``BasicImage``?
class CleanImage(Image):
    """
    Class that represents image made using CLEAN algorithm.
    """
    def __init__(self):
        super(CleanImage, self).__init__()
        self._beam = CleanBeam()
        # FIXME: Make ``_residuals`` a 2D-array only. Don't need ``Image``
        # instance
        self._residuals = None
        self._image_original = None
        self._name = None

    def _construct(self, **kwargs):
        """
        :param bmaj:
            Beam major axis [rad].
        :param bmin:
            Beam minor axis [rad].
        :param bpa:
            Beam positional angle [deg].
        :return:
        """
        super(CleanImage, self)._construct(**kwargs)
        # TODO: What if pixsize has different sizes???
        # FIXME: Beam has image twice the imsize. It's bad for plotting...
        kwargs["bmaj"] /= abs(kwargs["pixsize"][0])
        kwargs["bmin"] /= abs(kwargs["pixsize"][0])
        self._name = kwargs["name"]
        self._beam._construct(**kwargs)
        self._residuals = np.zeros(self.imsize, dtype=float)
        self._image_original = np.zeros(self.imsize, dtype=float)

    # def from_hdulist(self, hdulist, ver=1):
    #     super(CleanImage, self).from_hdulist(hdulist)

    #     model = Model(stokes=self.stokes)
    #     model.from_hdulist(hdulist, ver=ver)
    #     self.add_model(model)

    #     self._residuals = self._image_original - self.cc_image

    def __eq__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (super(CleanImage, self).__eq__(other) and
                self._beam.__eq__(other._beam))

    def __ne__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (super(CleanImage, self).__ne__(other) or
                self._beam.__ne__(other._beam))

    @property
    def beam_image(self):
        """
        Shorthand for beam image.
        """
        return self._beam.image

    @property
    def name(self):
        """
        Shorthand for source name.
        """
        return self._name

    @property
    def beam(self):
        """
        Shorthand for beam parameters bmaj [mas], bmin [mas], bpa [rad].
        """
        return (self._beam.beam[0] * abs(self.pixsize[0]) / mas_to_rad,
                self._beam.beam[1] * abs(self.pixsize[0]) / mas_to_rad,
                self._beam.beam[2])

    @beam.setter
    def beam(self, beam_pars):
        """
        Set beam parameters.

        :param beam_pars:
            Iterable of bmaj [mas], bmin [mas], bpa [deg].
        """
        # FIXME: Here i create new instance. Should i implement setter for
        # beam parameters in ``CleanBeam``?
        self._beam = CleanBeam()
        self._beam._construct(bmaj=beam_pars[0]*mas_to_rad/abs(self.pixsize[0]),
                              bmin=beam_pars[1]*mas_to_rad/abs(self.pixsize[0]),
                              bpa=beam_pars[2], imsize=self.imsize)

    @property
    def image(self):
        """
        Shorthand for clean components convolved with original clean beam with
        residuals added (that is what ``Image.image`` contains).
        """
        return self._image_original

    @image.setter
    def image(self, image):
        if isinstance(image, Image):
            if self == image:
                self._image_original = image.image.copy()
            else:
                raise Exception("Images have incompatible parameters!")
        # If ``image`` is array-like
        else:
            image = np.atleast_2d(image).copy()
            if not self.imsize == np.shape(image):
                raise Exception("Images have incompatible parameters!")
            self._image_original = image

    # def __mul__(self, other):
    #     if isinstance(other, CleanImage):
    #         if self == other:
    #             self.image *= other.image
    #         else:
    #             raise Exception("Different image parameters")
    #     else:
    #         self.image = self.image * other
    #     return self

    @property
    def cc_image(self):
        """
        Shorthand for convolved clean components image.
        """
        return signal.fftconvolve(self._image, self.beam_image, mode='same')

    @property
    def cc(self):
        """
        Shorthand for image of clean components (didn't convolved with beam).
        """
        return self._image

    @property
    def total_flux(self):
        return np.sum(self.cc)

    # FIXME: Should be read-only as residuals have sense only for naitive clean
    @property
    def residuals(self):
        return self._residuals

    def plot(self, to_plot, blc=None, trc=None, color_clim=None, cmap=None,
             abs_levels=None, rel_levels=None, min_abs_level=None,
             min_rel_level=None, factor=2., plot_color=False):
        """
        Plot image.

        :param to_plot:
            "cc", "ccr", "ccrr", "r" or "beam" - to plot only CC, CC Restored
            with beam, CC Restored with Residuals added, Residuals only or Beam.

        :note:
            ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
            converted to python-like zero-indexing.

        """
        plot_dict = {"cc": self._image, "ccr": self.image, "ccrr":
            self.image_w_residuals, "r": self._residuals.image,
                     "beam": self.beam}
        if plot_color:
            colors = plot_dict[to_plot]
            contours = None
        else:
            colors = None
            contours = plot_dict[to_plot]
        plot(contours, colors, x=self.x, y=self.y, blc=blc, trc=trc,
             color_clim=color_clim, cmap=cmap, abs_levels=abs_levels,
             rel_levels=rel_levels, min_abs_level=min_abs_level,
             min_rel_level=min_rel_level, k=factor)


def get_transverse_profile(ccfits, PA, nslices=200, plot_zobs_min=0, plot_zobs_max=None, beam=None, pixsize_mas=None,
                           treat_as_numpy_array=False, save_dir=None, save_prefix=None, save_figs=True, fig=None,
                           alpha=1.0, n_good_min=10, fig_res=None):
    from scipy.ndimage import rotate
    from astropy.stats import mad_std
    from astropy.modeling import fitting
    from astropy.modeling.models import custom_model, Gaussian1D
    from astropy import units as u

    if save_dir is None:
        save_dir = os.getcwd()
    if save_prefix is None:
        save_prefix = "transverse_profiles"

    if not treat_as_numpy_array:
        ccimage = create_clean_image_from_fits_file(ccfits)
        pixsize_mas = abs(ccimage.pixsize[0])*u.rad.to(u.mas)
        beam = ccimage.beam
        print("Beam (mas) : ", beam)
        image = ccimage.image
    else:
        image = ccfits

    size = image.shape[0]
    delta = round(size/2/nslices)
    print("Pixsize = {:.2f} mas".format(pixsize_mas))
    # Make jet directing down when plotting with origin=lower in matshow
    std = mad_std(image)
    print("std = {:.2f} mJy/beam".format(1000*std))
    image = rotate(image, PA, reshape=False)
    widths_mas = list()
    pos_mas = list()
    for i in range(nslices):
        imslice = image[int(size/2) - delta*i, :]
        g_init = Gaussian1D(amplitude=np.max(imslice), mean=size/2, stddev=beam[0]/pixsize_mas, fixed={'mean': True})
        fit_g = fitting.LevMarLSQFitter()
        x = np.arange(size)
        y = imslice
        mask = imslice > 5*std
        n_good = np.count_nonzero(mask)
        print("Number of unmasked elements for z = {:.2f} is N = {}".format(delta*i*pixsize_mas, n_good))
        if n_good < n_good_min:
            continue
        g = fit_g(g_init, x[mask], y[mask], weights=1/std)
        print("Convolved FWHM = {:.2f} mas".format(g.fwhm*pixsize_mas))
        width_mas_deconvolved = np.sqrt((g.fwhm*pixsize_mas)**2 - beam[0]**2)
        print("Deconvolved FWHM = {:.2f} mas".format(width_mas_deconvolved))
        if np.isnan(width_mas_deconvolved):
            continue
        widths_mas.append(width_mas_deconvolved)
        pos_mas.append(delta*i*pixsize_mas)

    pos_mas = np.array(pos_mas)
    widths_mas = np.array(widths_mas)
    if fig is None:
        fig, axes = plt.subplots(1, 1)
    else:
        axes = fig.get_axes()[0]
    if plot_zobs_max is not None:
        assert plot_zobs_max > plot_zobs_min
        axes.set_xlim([plot_zobs_min, plot_zobs_max])
        mask = np.logical_and(pos_mas < plot_zobs_max, pos_mas > plot_zobs_min)
        pos_to_plot = pos_mas[mask]
        widths_to_plot = widths_mas[mask]
    else:
        widths_to_plot = widths_mas
        pos_to_plot = pos_mas

    axes.plot(pos_to_plot, widths_to_plot, color="C0", alpha=alpha)
    axes.set_xlabel(r"$z_{\rm obs}$, mas")
    axes.set_ylabel("FWHM, mas")
    plt.xscale("log")
    plt.yscale("log")
    if save_figs:
        fig.savefig(os.path.join(save_dir, "{}.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    # Now fit profile
    @custom_model
    def power_law(r, amp=1.0, r0=0.0, k=0.5):
        return amp*(r + r0)**k
    pl_init = power_law(fixed={"r0": True})
    fit_pl = fitting.LevMarLSQFitter()
    pl = fit_pl(pl_init, pos_to_plot, widths_to_plot, maxiter=10000)
    print(fit_pl.fit_info)
    print("k = ", pl.k)
    print("r0 = ", pl.r0)
    print("amp = ", pl.amp)

    # Plot fit
    xx = np.linspace(np.min(pos_to_plot), np.max(pos_to_plot), 1000)
    yy = pl(xx)
    fig_, axes = plt.subplots(1, 1)
    axes.plot(xx, yy, color="C1", label="k = {:.2f}".format(pl.k.value))
    axes.scatter(pos_to_plot, widths_to_plot, color="C0", label="data", s=2)
    axes.set_xlabel(r"$z_{\rm obs}$, mas")
    axes.set_ylabel("FWHM, mas")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    if save_figs:
        fig_.savefig(os.path.join(save_dir, "{}_fit.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig_)

    # Make residuals and plot them
    res = widths_to_plot - pl(pos_to_plot)
    max_res = 1.2*np.max(np.abs(res))

    if fig_res is None:
        fig_res, axes = plt.subplots(1, 1)
    else:
        axes = fig_res.get_axes()[0]

    axes.plot(pos_to_plot, res, color="C0", alpha=1.0)
    # axes.set_ylim([-max_res, max_res])
    axes.set_ylim([-2, 2])
    axes.set_xlabel(r"$z_{\rm obs}$, mas") 
    axes.set_ylabel("residual FWHM, mas")
    if save_figs:
        fig_res.savefig(os.path.join(save_dir, "{}_residual_width.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    return fig, fig_res


class Jet_data(CleanImage):

    def __init__(self):
        super().__init__()
        self._image_dict = {}

    def from_hdulist(self, hdulist, set_stokes=None):
        image_params = get_fits_image_info_from_hdulist(hdulist)
        if self._name is not None and self._name != image_params["name"]:
            raise Exception("Observations of different sources are loaded!")
        self._construct(**image_params)
        pr_hdu = get_hdu_from_hdulist(hdulist)
        if set_stokes is None:
            set_stokes = image_params["stokes"]
        self._image_dict[set_stokes] = pr_hdu.data.squeeze()

    def from_fits(self, fname, set_stokes=None):
        hdulist = pf.open(fname)
        self.from_hdulist(hdulist, set_stokes=set_stokes)

    @property
    def image_dict(self):
        return self._image_dict
    
    @property
    def image(self):
        warnings.warn("This method isnt avalable in this class!")
        pass
    
    @property
    def stokes(self):
        return list(self._image_dict.keys())
    
    @stokes.setter
    def stokes(self, _):
        warnings.warn("This method isnt avalable in this class!")
        pass

    def get_image(self, stk=None):
        if len(self._image_dict) == 0:
            raise Exception(f"No image data found! Use .from_fits method")
        if stk is None:
            if len(self._image_dict) == 1:
                stk = list(self._image_dict.keys())[0]
            else:
                stk = "I"
        stk = stk.upper()
        if stk == "P":
            if "Q" in self._image_dict and "U" in self._image_dict:
                return np.hypot(self._image_dict["U"], self._image_dict["Q"])
            else:
                raise Exception("Load polarisation images!")
        elif stk == "CHI":
            if "Q" in self._image_dict and "U" in self._image_dict:
                return 0.5*np.arctan2(self._image_dict["U"], self._image_dict["Q"])
            else:
                raise Exception("Load polarisation images!")
        elif stk == "F":
            if "Q" in self._image_dict and "U" in self._image_dict and "I" in self._image_dict:
                return np.hypot(self._image_dict["U"], self._image_dict["Q"])/self._image_dict["I"]
            else:
                raise Exception("Load polarisation and I-stokes images!")
        elif stk in self._image_dict:
            return self._image_dict[stk]
        else:
            raise Exception(f"Unknown stokes! Avalible: {list(self._image_dict.keys())}")

    def slice(self, pix1=None, pix2=None, point1=None, point2=None):
        """
        Return slice of image in pixels or image coordinates.
        :param pix1: (optional)
            Iterable of coordinates of first pixel [pixels].
        :param pix2: (optional)
            Iterable of coordinates of second pixel [pixels].
        :param point1: (optional)
            Iterable of coordinates of first pixel [mas].
        :param point2: (optional)
            Iterable of coordinates of second pixel [mas].
        :return:
            Numpy array of image values for given slice.
        """
        if point1 is not None and point2 is not None:
            # Convert coordinates [mas] to pixels
            pix1, pix2 = self._convert_coordinates(point1, point2)

        length = int(round(np.hypot(pix2[0] - pix1[0], pix2[1] - pix1[1])))
        if pix2[0] < pix1[0]:
            x = np.linspace(pix2[0], pix1[0], length)[::-1]
        else:
            x = np.linspace(pix1[0], pix2[0], length)
        if pix2[1] < pix1[1]:
            y = np.linspace(pix2[1], pix1[1], length)[::-1]
        else:
            y = np.linspace(pix1[1], pix2[1], length)

        sl = {"ra":[], "dec":[]}
        for x_, y_ in zip(x, y):
            dec, ra = self._convert_array_coordinate((int(round(x_)), int(round(y_))))
            sl["dec"].append(dec)
            sl["ra"].append(ra)
        for stk in self.stokes:
            sl[stk] = self.get_image(stk=stk)[v_round(y).astype(int), v_round(x).astype(int)]
        return sl


#class MemImage(BasicImage, Model):
#    """
#    Class that represents image made using MEM algorithm.
#    """
#    pass

if __name__ == '__main__':
    import matplotlib
    from astropy.stats import mad_std
    from from_fits import create_clean_image_from_fits_file

    # ccimage = "/home/ilya/data/alpha/results/MOJAVE/model_cc_i_15.4.fits"
    # ccimage = create_clean_image_from_fits_file(ccimage)
    # beam = ccimage.beam
    # print(beam)
    # std = mad_std(ccimage.image)
    # print(std)
    # blc = (400, 430)
    # trc = (980, 710)
    # print(blc, trc)
    # matplotlib.use('Qt5Agg')
    #
    # figsize = (12, 10)
    # fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=True, sharex=True)
    # plt.subplots_adjust(hspace=0, wspace=0)
    # plot(ccimage.image, ccimage.image, x=ccimage.x, y=ccimage.y,
    #      min_abs_level=3*std, colors_mask=ccimage.image < 5*std,
    #      color_clim=None, blc=blc, trc=trc,
    #      beam=beam, colorbar_label=r"$I$, Jy/beam", show_beam=True,
    #      cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
    #      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
    #      axes=axes[0], show_xlabel_on_current_axes=False, show_ylabel_on_current_axes=True)
    # plot(ccimage.image, ccimage.image, x=ccimage.x, y=ccimage.y,
    #      min_abs_level=3*std, colors_mask=ccimage.image < 5*std,
    #      color_clim=None, blc=blc, trc=trc,
    #      beam=beam, colorbar_label=r"$I$, Jy/beam", show_beam=True,
    #      cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
    #      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
    #      axes=axes[1], show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True)
    # plt.show()


    # Make transverse slice
    label_size = 22
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    from scipy.ndimage import rotate6
    # image = pf.getdata("model_cc_i_15.4_0.12pc.fits").squeeze()
    # R = 0.09
    image = pf.getdata("/home/ilya/Downloads/model_cc_i_15.4.fits").squeeze()
    size = image.shape[0]
    image = rotate(image, 17, reshape=False)

    # from 20 to 30 mas
    im = [image[int(size/2)-50:int(size/2)+50, int(size/2)+int(i)] for i in np.linspace(200, 300, 50)]
    low, med, up = scoreatpercentile(im, [16, 50, 84], axis=0)
    std = np.std(im, axis=0)
    mean = np.mean(im, axis=0)
    # -5 to 5 mas
    x = np.linspace(-5, 5, 100)

    fig, axes = plt.subplots(1, 1, figsize=(13, 4))
    axes.plot(x, 1000*mean, lw=4, label="stack")
    axes.fill_between(x, 1000*(mean-std), 1000*(mean+std), alpha=0.3, color="C0", label="std")
    axes.set_xlabel(r"Transverse distance, mas")
    axes.set_ylabel(r"Intensity, mJy/beam")
    # axes.set_xlim([-5, 5])
    # 0.09 pc
    # axes.axhline(y=1.5, xmin=0.1, xmax=0.1+0.86/10, lw=5, color="black")
    # 0.12 pc
    axes.axhline(y=0.4, xmin=0.1, xmax=0.1+0.86/10, lw=5, color="black")
    # axes.set_ylim([0, 1.7])
    # 0.09 pc
    # axes.text(0.09, 0.78, "FWHM:", fontdict={"fontsize": 22}, transform=axes.transAxes, ha="left")
    # 0.12 pc
    axes.text(0.09, 0.68, "FWHM:", fontdict={"fontsize": 22}, tra850nsform=axes.transAxes, ha="left")
    plt.legend(loc="upper right")
    plt.savefig("/home/ilya/Documents/EVN2022/slice_mean_20_30_mas_0.09pc_wide.png", bbox_inches="tight", dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    axes.plot(x, 1000*med, lw=4)
    axes.fill_between(x, 1000*low, 1000*up, alpha=0.3, color="C0")
    axes.set_xlabel(r"$R$, mas")
    axes.set_ylabel(r"$I$, mJy/beam")
    # plt.savefig("/home/ilya/Documents/EVN2022/slice_med_std_20_30_mas_0.09pc.png", dpi=300)
    plt.show()