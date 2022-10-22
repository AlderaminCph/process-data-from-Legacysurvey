"""
Cut out galaxy image from fits file

>>> get_galaxy_diameter('ESO545-040')
94.9
"""
import doctest
import pandas as pd
import astropy.io.fits as fits
from typing import Tuple
import numpy as np


def get_galaxy_diameter(
    galaxyname: str, filename: str = "S0.20220728.dat"
) -> float:
    """Retrive the diameter of the galaxy by its name from the sample dat file
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
    """Calculates pixel coordinates of the fits file center
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
    """Retrive the image data from fits file
    Args:
        fits_file: full path to the fits file
    Return:
        numpy ndarray with image data values
    """
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
    return image_data


doctest.testmod()
