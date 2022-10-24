"""
Download fits images and invariance maps of the galaxies from Leda survey
by its name.

Example of usage:
python3 download_fits_images.py 'ESO545-040'

"""
import os
import sys

from astropy.io import fits
from crop_fits import get_galaxy_ra_dec, read_sample_file

if __name__ == "__main__":
    galaxyname = sys.argv[1]
    df = read_sample_file()
    print(df[df["objname"] == galaxyname])
    cel_coord = get_galaxy_ra_dec(galaxyname, df)
    ra = cel_coord.ra.degree
    dec = cel_coord.dec.degree
    print(ra, dec)

    top_level_directory = (
        "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/"
    )
    dirname = os.getcwd()
    print("DIRNAME", dirname)
    fits_tables_directory = os.path.join(dirname, "/fits_tables")
    print("FITS TABLE DIRECTORY", fits_tables_directory)
    if not os.path.isfile("survey-bricks.fits.gz") and not os.path.isfile(
        "survey-bricks.fits"
    ):
        w_cmd = "wget " + top_level_directory + "survey-bricks.fits.gz"
        print("COMMAND ", w_cmd)
        os.system(w_cmd)
    if not os.path.isfile("survey-bricks.fits"):
        os.system("gzip -d " + "survey-bricks.fits.gz")

    with fits.open("survey-bricks.fits") as hdul:
        data = hdul[1].data

    mask = (
        (data["RA1"] <= ra)
        & (data["RA2"] >= ra)
        & (data["DEC1"] <= dec)
        & (data["DEC2"] >= dec)
    )
    brick = str(data[mask]["BRICKNAME"][0])
    print("BRICK ", brick)
    ra_dir_num, _ = divmod(ra, 1)
    ra_dir_num = "0" + str(ra_dir_num).split(".")[0]
    print("AAA", ra_dir_num)
    file_directory_url = (
        top_level_directory + "south/coadd/" + ra_dir_num + "/" + brick + "/"
    )
    file_name_pattern = galaxyname.replace(" ", "")
    file_name_pattern = file_name_pattern.replace("-", "_")

    galaxy_fits_dir = (
        "/home/alderamin/Desktop/GRANT_WORK/process_data/galaxy_fits/"
    )
    fitsfile = os.path.join(galaxy_fits_dir, file_name_pattern + "_g.fits")

    if not os.path.isfile(fitsfile):
        w_cmd = (
            "wget "
            + file_directory_url
            + "legacysurvey-"
            + brick
            + "-image-g.fits.fz"
            + " -O "
            + file_name_pattern
            + "_g.fits"
        )
        print("COMMAND ", w_cmd)
        os.system(w_cmd)
        move = "mv " + file_name_pattern + "_g.fits " + galaxy_fits_dir
        os.system(move)

    invvar_fits_dir = (
        "/home/alderamin/Desktop/GRANT_WORK/process_data/invariance_maps"
    )
    invarfile = os.path.join(
        invvar_fits_dir,
        file_name_pattern + "_invar_g.fits",
    )

    if not os.path.isfile(invarfile):
        w_cmd = (
            "wget "
            + file_directory_url
            + "legacysurvey-"
            + brick
            + "-invvar-g.fits.fz"
            + " -O "
            + file_name_pattern
            + "_invvar_g.fits"
        )
        print("COMMAND ", w_cmd)
        os.system(w_cmd)
        move = "mv " + file_name_pattern + "_invvar_g.fits " + invvar_fits_dir
        os.system(move)
