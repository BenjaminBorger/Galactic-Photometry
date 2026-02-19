from astropy.io import fits

with fits.open('Mosaic_Draco2_2020_0188_0217_M100.fits') as hdul:
    hdul.info()  # Shows structure
    print(repr(hdul[0].header))  # Prints the header