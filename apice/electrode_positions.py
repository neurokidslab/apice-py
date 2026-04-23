"""Electrode and head-shape geometry helpers.

This module provides sphere-fitting utilities derived from MNE internals and
used by APICE interpolation and spatial processing routines.
"""

import glob
import json
import os
import os.path as op
import shutil
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import fmin_cobyla

from mne._fiff._digitization import _dig_kind_dict, _dig_kind_ints, _dig_kind_rev
from mne._fiff.constants import FIFF, FWD
from mne._fiff.open import fiff_open
from mne._fiff.tag import find_tag
from mne._fiff.tree import dir_tree_find
from mne._fiff.write import (
    end_block,
    start_and_end_file,
    start_block,
    write_float,
    write_float_matrix,
    write_int,
    write_int_matrix,
    write_string,
)
from mne.fixes import _compare_version, _safe_svd
from mne.surface import (
    _complete_sphere_surf,
    _compute_nearest,
    _fast_cross_nd_sum,
    _get_ico_surface,
    _get_solids,
    complete_surface_info,
    decimate_surface,
    read_surface,
    read_tri,
    transform_surface_to,
    write_surface,
)
from mne.transforms import Transform, _ensure_trans, apply_trans
from mne.utils import (
    _check_fname,
    _check_freesurfer_home,
    _check_head_radius,
    _check_option,
    _ensure_int,
    _import_h5io_funcs,
    _import_nibabel,
    _on_missing,
    _path_like,
    _pl,
    _TempDir,
    _validate_type,
    _verbose_safe_false,
    get_subjects_dir,
    logger,
    path_like,
    run_subprocess,
    verbose,
    warn,
)
from mne.viz.misc import plot_bem

def fit_sphere_to_headshape(info, dig_kinds="auto", units="m", verbose=None):
    """Fit a sphere to digitized head-shape points.

    Parameters
    ----------
    info : mne.Info
        Measurement info containing digitization points in ``info['dig']``.
    dig_kinds : str | tuple, default='auto'
        Digitization point kinds to include in the fit.
    units : str
        Can be ``"m"`` (default) or ``"mm"``.

        .. versionadded:: 0.12
    verbose : bool | str | int | None
        MNE verbosity control.

    Returns
    -------
    radius : float
        Fitted sphere radius in the requested units.
    origin_head : ndarray, shape (3,)
        Sphere origin in head coordinates.
    origin_device : ndarray, shape (3,)
        Sphere origin in device/MEG coordinates.

    Notes
    -----
    This function excludes any points that are low and frontal
    (``z < 0 and y > 0``) to improve the fit.
    """
    if not isinstance(units, str) or units not in ("m", "mm"):
        raise ValueError('units must be a "m" or "mm"')
    radius, origin_head, origin_device = _fit_sphere_to_headshape(info, dig_kinds)
    if units == "mm":
        radius *= 1e3
        origin_head *= 1e3
        origin_device *= 1e3
    return radius, origin_head, origin_device

def get_fitting_dig(info, dig_kinds="auto", exclude_frontal=True, verbose=None):
    """Return digitization points suitable for sphere fitting.

    Parameters
    ----------
    info : mne.Info
        Measurement info containing digitization points.
    dig_kinds : str | tuple, default='auto'
        Point types to include.
    exclude_frontal : bool, default=True
        If True, remove low frontal points likely belonging to the face/nose.

        .. versionadded:: 0.19
    verbose : bool | str | int | None
        MNE verbosity control.

    Returns
    -------
    dig : ndarray, shape (n_points, 3)
        Selected digitization points in head coordinates.

    Notes
    -----
    This will exclude digitization locations that have ``z < 0 and y > 0``,
    i.e. points on the nose and below the nose on the face.

    .. versionadded:: 0.14
    """
    _validate_type(info, "info")
    if info["dig"] is None:
        raise RuntimeError(
            'Cannot fit headshape without digitization, info["dig"] is None'
        )
    if isinstance(dig_kinds, str):
        if dig_kinds == "auto":
            # try "extra" first
            try:
                return get_fitting_dig(info, "extra")
            except ValueError:
                pass
            return get_fitting_dig(info, ("extra", "eeg"))
        else:
            dig_kinds = (dig_kinds,)
    # convert string args to ints (first make dig_kinds mutable in case tuple)
    dig_kinds = list(dig_kinds)
    for di, d in enumerate(dig_kinds):
        dig_kinds[di] = _dig_kind_dict.get(d, d)
        if dig_kinds[di] not in _dig_kind_ints:
            raise ValueError(
                f"dig_kinds[{di}] ({d}) must be one of {sorted(_dig_kind_dict)}"
            )

    # get head digization points of the specified kind(s)
    dig = [p for p in info["dig"] if p["kind"] in dig_kinds]
    if len(dig) == 0:
        raise ValueError(f"No digitization points found for dig_kinds={dig_kinds}")
    if any(p["coord_frame"] != FIFF.FIFFV_COORD_HEAD for p in dig):
        raise RuntimeError(
            f"Digitization points dig_kinds={dig_kinds} not in head "
            "coordinates, contact mne-python developers"
        )
    hsp = [p["r"] for p in dig]
    del dig

    # exclude some frontal points (nose etc.)
    if exclude_frontal:
        hsp = [p for p in hsp if not (p[2] < -1e-6 and p[1] > 1e-6)]
    hsp = np.array(hsp)

    if len(hsp) <= 10:
        kinds_str = ", ".join([f'"{_dig_kind_rev[d]}"' for d in sorted(dig_kinds)])
        msg = (
            f"Only {len(hsp)} head digitization points of the specified "
            f"kind{_pl(dig_kinds)} ({kinds_str},)"
        )
        if len(hsp) < 4:
            raise ValueError(msg + ", at least 4 required")
        else:
            warn(msg + ", fitting may be inaccurate")
    return hsp

def _fit_sphere_to_headshape(info, dig_kinds, verbose=None):
    """Internal helper that fits a sphere and computes coordinate transforms.

    Parameters
    ----------
    info : mne.Info
        Measurement info with digitization points and device-head transform.
    dig_kinds : str | tuple
        Digitization point kinds to include in the fit.
    verbose : bool | str | int | None
        MNE verbosity control.

    Returns
    -------
    radius : float
        Fitted sphere radius in meters.
    origin_head : ndarray, shape (3,)
        Sphere origin in head coordinates.
    origin_device : ndarray, shape (3,)
        Sphere origin transformed into device coordinates.
    """
    hsp = get_fitting_dig(info, dig_kinds)
    radius, origin_head = _fit_sphere(np.array(hsp), disp=False)
    # compute origin in device coordinates
    dev_head_t = info["dev_head_t"]
    if dev_head_t is None:
        dev_head_t = Transform("meg", "head")
    head_to_dev = _ensure_trans(dev_head_t, "head", "meg")
    origin_device = apply_trans(head_to_dev, origin_head)
    #logger.info("Fitted sphere radius:".ljust(30) + f"{radius * 1e3:0.1f} mm")
    _check_head_radius(radius)

    # > 2 cm away from head center in X or Y is strange
    o_mm = origin_head * 1e3
    o_d = origin_device * 1e3
    if np.linalg.norm(origin_head[:2]) > 0.02:
        warn(
            f"(X, Y) fit ({o_mm[0]:0.1f}, {o_mm[1]:0.1f}) "
            "more than 20 mm from head frame origin"
        )
    """ logger.info(
        "Origin head coordinates:".ljust(30)
        + f"{o_mm[0]:0.1f} {o_mm[1]:0.1f} {o_mm[2]:0.1f} mm"
    )
    logger.info(
        "Origin device coordinates:".ljust(30)
        + f"{o_d[0]:0.1f} {o_d[1]:0.1f} {o_d[2]:0.1f} mm"
    ) """
    return radius, origin_head, origin_device

def _fit_sphere(points, disp="auto"):
    """Fit a sphere to 3D points using constrained optimization.

    Parameters
    ----------
    points : ndarray, shape (n_points, 3)
        Coordinates used for sphere fitting.
    disp : bool | str, default='auto'
        Optimization display mode for ``scipy.optimize.fmin_cobyla``.

    Returns
    -------
    radius : float
        Fitted sphere radius.
    origin : ndarray, shape (3,)
        Fitted sphere origin.
    """
    if isinstance(disp, str) and disp == "auto":
        disp = True if logger.level <= 20 else False
    # initial guess for center and radius
    radii = (np.max(points, axis=1) - np.min(points, axis=1)) / 2.0
    radius_init = radii.mean()
    center_init = np.median(points, axis=0)

    # optimization
    x0 = np.concatenate([center_init, [radius_init]])

    def cost_fun(center_rad):
        d = np.linalg.norm(points - center_rad[:3], axis=1) - center_rad[3]
        d *= d
        return d.sum()

    def constraint(center_rad):
        return center_rad[3]  # radius must be >= 0

    x_opt = fmin_cobyla(
        cost_fun,
        x0,
        constraint,
        rhobeg=radius_init,
        rhoend=radius_init * 1e-6,
        disp=disp,
    )

    origin, radius = x_opt[:3], x_opt[3]
    return radius, origin


def _check_origin(origin, info, coord_frame="head", disp=False):
    """Validate or infer an origin point in a given coordinate frame.

    Parameters
    ----------
    origin : array-like | str
        Explicit 3-element origin or ``'auto'`` to estimate it.
    info : mne.Info
        Measurement info used when ``origin='auto'``.
    coord_frame : {'head', 'meg'}, default='head'
        Coordinate frame for the returned origin.
    disp : bool, default=False
        If True, logs the chosen origin.

    Returns
    -------
    origin : ndarray, shape (3,)
        Validated origin in the requested coordinate frame.

    Raises
    ------
    ValueError
        If ``origin`` is not a valid 3-element coordinate specification.
    """
    if isinstance(origin, str):
        if origin != "auto":
            raise ValueError(
                f'origin must be a numerical array, or "auto", not {origin}'
            )
        if coord_frame == "head":
            R, origin = fit_sphere_to_headshape(
                info, verbose=_verbose_safe_false(), units="m"
            )[:2]
            #logger.info(f"    Automatic origin fit: head of radius {R * 1000:0.1f} mm")
            del R
        else:
            origin = (0.0, 0.0, 0.0)
    origin = np.array(origin, float)
    if origin.shape != (3,):
        raise ValueError("origin must be a 3-element array")
    if disp:
        origin_str = ", ".join([f"{o * 1000:0.1f}" for o in origin])
        msg = f"    Using origin {origin_str} mm in the {coord_frame} frame"
        if coord_frame == "meg" and info["dev_head_t"] is not None:
            o_dev = apply_trans(info["dev_head_t"], origin)
            origin_str = ", ".join(f"{o * 1000:0.1f}" for o in o_dev)
            msg += f" ({origin_str} mm in the head frame)"
        logger.info(msg)
    return origin
