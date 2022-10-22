"""
Cut out galaxy image from fits file.

>>> get_galaxy_diameter('ESO545-040')
94.9
"""
import doctest
from typing import Tuple

import astropy.io.fits as fits
import astropy.wcs as wcs
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D


def get_galaxy_diameter(
    galaxyname: str, filename: str = "S0.20220728.dat"
) -> float:
    """Retrive the diameter of the galaxy by its name from the sample dat file.

    Args:
        filename: the dat file with the infotmation about the sample
        galaxyname: the object name from dat file
    Return:
        galaxy_diameter: the diameter in arcseconds
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
    galaxy_diameter = float(df[df["objname"] == galaxyname]["d25arcsec"])
    return galaxy_diameter


def get_central_pix_coordinates(fits_file: str) -> Tuple[int, int]:
    """Calculate pixel coordinates of the fits file center.

    Args:
        fits_file: full path to the fits file
    Return:
        tuple (x_0, y_0)
    """
    with fits.open(fits_file) as hdul:
        hdr = hdul[0].header
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
        return hdul[0].data


def get_wcs_from_file(fits_file: str) -> wcs.WCS:
    """Retrive wcs from fits file.

    Args:
        fits_file: full path to the fits file
    Return:
        wcs object
    """
    with fits.open(fits_file) as hdul:
        return wcs.WCS(hdul[0].header, hdul)


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
        diameter_pix: the diameter of the galaxy in arcseconds
        w: wcs object of fits file
    """
    cutout = Cutout2D(
        image_data,
        position=pos,
        size=(2 * diameter_pix, 2 * diameter_pix),
        wcs=w,
    )
    hdu = fits.PrimaryHDU()
    hdu.data = cutout.data
    hdu.header.update(cutout.wcs.to_header())
    hdu.writeto(cutout_name)


doctest.testmod()
