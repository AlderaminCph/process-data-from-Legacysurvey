"""
Sypkova Anastasia

Cut out rectangular path with galaxy image from fits file with the size of 3
diameters of galaxy

Example of usage:
python3 crop_files.py 'ESO545-040'

>>> df = read_sample_file(os.path.join(\
"~/Desktop/GRANT_WORK/process_data/code", "S0.20220728.dat"))
>>> get_galaxy_diameter('ESO545-040', df)
94.9

>>> get_galaxy_ra_dec('ESO545-040', df).ra.degree
39.54879

>>> get_galaxy_ra_dec('ESO545-040', df).dec.degree
-20.16696

"""
import doctest
import os
import sys
from typing import Optional, Tuple

import astropy.io.fits as fits
import astropy.wcs as wcs
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    PowerStretch,
)
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from matplotlib import colormaps


def read_sample_file(
    filename: str = "S0.20220728.dat",
) -> pd.core.frame.DataFrame:
    """Read data from the file containing information about galaxy sample and
     creates dataframe with it.

    Args:
        filename: the dat file with the infotmation about the sample
    Return:
        df: pandas dataframe
    """
    df = pd.read_csv(
        filename, sep="|", header=0, skiprows=lambda x: x in [1, 44501]
    )
    df.columns = df.columns.str.strip()
    for col in df:
        if isinstance(df[col].iloc[df[col].first_valid_index()], str):
            df[col] = df[col].str.strip()
        if isinstance(df[col].iloc[df[col].first_valid_index()], float):
            df[col] = df[col].astype("float")
    return df


def get_galaxy_diameter(galaxyname: str, df: pd.core.frame.DataFrame) -> float:
    """Retrive the galaxy diameter by its name from the sample pd.dataframe.

    Args:
        galaxyname: the object name from dat file
    Return:
        galaxy_diameter: the diameter in arcseconds
    """
    return float(df[df["objname"] == galaxyname]["d25arcsec"])


def get_galaxy_ra_dec(
    galaxyname: str, df: pd.core.frame.DataFrame
) -> Optional[SkyCoord]:
    """Retrive the galaxy celestial coordinates from the sample dat file.

    Args:
        galaxyname: the name of the galaxy
        df: pandas dataframe with the galaxy sample information
    Return:
        cel_coord: pair of values ra and dec of galaxy
    """
    ra = float(df[df["objname"] == galaxyname]["ra"])
    dec = float(df[df["objname"] == galaxyname]["de"])
    cel_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    return cel_coord


def get_central_pix_coordinates(fits_file: str) -> Tuple[int, int]:
    """Calculate pixel coordinates of the fits file center.

    Args:
        fits_file: full path to the fits file
    Return:
        tuple (x_0, y_0)
    """
    with fits.open(fits_file) as hdul:
        hdr = hdul[1].header
        x_0 = int(hdr["NAXIS1"] / 2)
        y_0 = int(hdr["NAXIS2"] / 2)
    return (x_0, y_0)


def get_image_data(fits_file: str) -> np.ndarray:
    """Retrive the image data from fits file.

    Args:
        fits_file: full path to the fits file
    Return:
        numpy ndarray with image data values
    """
    with fits.open(fits_file) as hdul:
        return hdul[1].data if len(hdul) > 1 else hdul[0].data


def get_wcs_from_file(fits_file: str) -> wcs.WCS:
    """Retrive wcs from fits file.

    Args:
        fits_file: full path to the fits file
    Return:
        wcs object
    """
    with fits.open(fits_file) as hdul:
        return (
            wcs.WCS(hdul[1].header, hdul)
            if len(hdul) > 1
            else wcs.WCS(hdul[0].header, hdul)
        )


def create_crop_fits(
    image_data: np.ndarray,
    pos: Tuple[int, int],
    diameter_pix: float,
    w: wcs.WCS,
    cutout_name: str,
) -> None:
    """Create a new fits sile containing the cutout image.

    Args:
        image_data: numpy ndarray with image data values
        pos: tuple (x_0, y_0) with pixel coordinates of the galaxy center
        diameter_pix: the diameter of the galaxy in pixels
        w: wcs object of fits file
    """
    if not os.path.isfile(cutout_name):

        cutout = Cutout2D(
            image_data,
            position=pos,
            size=(3 * diameter_pix, 3 * diameter_pix),
            wcs=w,
        )
        hdu = fits.PrimaryHDU()
        hdu.data = cutout.data
        hdu.header.update(cutout.wcs.to_header())
        hdu.writeto(cutout_name)


def get_pix_diameter(diameter_arcsec: float, w: wcs.WCS) -> float:
    """Convert the galaxy diameter from arcseconds to pixels.

    Args:
        diameter_arcsec: the galaxy diameter in arcseconds
        w: wcs object of fits file containing the galaxy image
    Return:
        diameter_pix: the galaxy diameter in pixels
    """
    scale = u.pixel_scale(
        proj_plane_pixel_scales(w)[0] * w.wcs.cunit[0] / u.pixel
    )
    diameter_pix = (diameter_arcsec * u.arcsec).to(u.pixel, scale).value
    return diameter_pix


def set_plt_parameters():
    """
    Set Latex plot parameters for slices pictures
    """

    plt.rc("text", usetex=True)
    plt.rc(
        "text.latex",
        preamble="\n".join(
            [r"\usepackage{amsmath,amssymb}", r"\usepackage[russian]{babel}"]
        ),
    )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 20,
        }
    )


def plot_crop_png(
    image_data: np.ndarray,
    w: wcs.WCS,
    pos_pix: Tuple[float, float],
    diameter_pix: float,
    galaxyname: str,
    pngname: str,
) -> None:
    """Plot png image of cutout fits with galaxy.

    Args:
        image_data: numpy ndarray with image data values
        w: wcs object of fits file containing the galaxy image
        pos_pix: pixel position (x_0, y_0) of the galaxy
        diameter_pix: the galaxy diameter in pixels
        galaxyname: name of the galaxy
        pngname: name of png file to save the picture
    """
    set_plt_parameters()
    fig, ax = plt.subplots(figsize=(8, 6))
    interval = PercentileInterval(99.1)
    vmin, vmax = interval.get_limits(image_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=PowerStretch(0.5))
    color = colormaps["inferno"]
    ax = plt.subplot(projection=w)
    ax.imshow(image_data, cmap=color, norm=norm, origin="lower")
    ax.grid(color="white", ls="dotted")
    ax.set_xlabel(r"$\alpha$", fontsize=25)
    ax.set_ylabel(r"$\delta$", fontsize=25)
    ax.set_title(galaxyname)

    # Create a Rectangle patch
    x_0, y_0 = pos_pix
    rect = patches.Rectangle(
        (x_0 - diameter_pix, y_0 - diameter_pix),
        3 * diameter_pix,
        3 * diameter_pix,
        linewidth=1,
        edgecolor="lightgreen",
        facecolor="none",
    )

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.savefig(pngname, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    galaxyname = sys.argv[1]
    df = read_sample_file(
        os.path.join(
            "~/Desktop/GRANT_WORK/process_data/code", "S0.20220728.dat"
        )
    )
    file_name_pattern = galaxyname.replace(" ", "")
    file_name_pattern = file_name_pattern.replace("-", "_")

    fits_file_image = os.path.join(
        "/home/alderamin/Desktop/GRANT_WORK/process_data/galaxy_fits",
        file_name_pattern + "_g.fits",
    )
    invar_file = os.path.join(
        "/home/alderamin/Desktop/GRANT_WORK/process_data/invariance_maps",
        file_name_pattern + "_invvar_g.fits",
    )
    for fits_file in [invar_file, fits_file_image]:
        print("FITSFILE ", fits_file)
        image_data = get_image_data(fits_file)
        fits_wcs = get_wcs_from_file(fits_file)
        print("WCS ", fits_wcs)
        cel_coord = get_galaxy_ra_dec(galaxyname, df)
        print("RA DEC", cel_coord)
        pos_pix = skycoord_to_pixel(cel_coord, fits_wcs)
        diameter_arcsec = get_galaxy_diameter(galaxyname, df)
        diameter_pix = get_pix_diameter(diameter_arcsec, fits_wcs)
        cut_directory = os.path.dirname(fits_file) + "/cut"
        cutout_name = os.path.join(
            cut_directory, "cut_" + fits_file.split("/")[-1]
        )
        create_crop_fits(
            image_data, pos_pix, diameter_pix, fits_wcs, cutout_name
        )

doctest.testmod()
