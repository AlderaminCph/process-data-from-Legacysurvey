"""
Cut out galaxy image from fits file

>>> get_galaxy_diameter('ESO545-040')
94.9
"""
import doctest
import pandas as pd

# from typing import Tuple


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


doctest.testmod()
