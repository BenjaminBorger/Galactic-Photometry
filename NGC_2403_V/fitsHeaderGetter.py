from astropy.io import fits
import parameters
import numpy as np
galaxyName,file, notimportant = parameters.names()
from scipy.ndimage import rotate

hdul = fits.open(file)
header = hdul[0].header
image = hdul[0].data

centrepix_X  = header['CRPIX1']
centrepix_Y = header['CRPIX2']
RA = header["RA"]
DEC = header["DEC"]

ps,zp = parameters.imageCalibration() # recall ps is pixel per arcsec



#coordinates in pixels
#change in pixels
#convert change of pixels into degrees 
#add or subtract degrees from ra and dec

def pixelsToDegrees(pixel):
    return (pixel*ps)/3600



def pixelCoordinatesToRADEC(starX_pix,starY_pix):

    differenceX_deg = pixelsToDegrees(centrepix_X-starX_pix)
    differenceY_deg = pixelsToDegrees(centrepix_Y-starY_pix)
    

    cd1_1 = header['CD1_1']
    cd1_2 = header['CD1_2']
    # Rotation angle in degrees
    angle = np.degrees(np.arctan2(cd1_2, cd1_1))

    rotated = rotate(image, angle, reshape=False, cval=np.nan)

    # Save result
    hdul[0].data = rotated

    RA_star = RA+differenceX_deg
    DEC_star = DEC+differenceY_deg
    

    return RA_star, DEC_star


print(pixelCoordinatesToRADEC(561.8928681,422.8842405))


'''
with fits.open(file) as hdul:
    hdul.info()  # Shows structure
    print(repr(hdul[0].header))  # Prints the header

'''
