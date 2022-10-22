"""
Sypkova Anastasia

Cut out galaxy image from fits file.

>>> df = read_sample_file("S0.20220728.dat")
>>> get_galaxy_diameter('ESO545-040', df)
94.9

>>> get_galaxy_ra_dec('ESO545-040', df).ra.degree
39.54879

>>> get_galaxy_ra_dec('ESO545-040', df).dec.degree
-20.16696

"""
import doctest
from typing import Optional, Tuple

import astropy.io.fits as fits
import astropy.wcs as wcs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs.utils import proj_plane_pixel_scales


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
        diameter_pix: the diameter of the galaxy in pixels
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


doctest.testmod()
