def names():
    
    inputFile = "Mosaic_Draco2_2020_0188_0217_M100.fits" #string with name of mosaiced galaxy to analyse 
    outputFile = 'surface_brightness_profileMosaicDraco2.dat' #string with name of where surface brightness profile should go
    return(inputFile,outputFile)

def galaxyInfo():
    
    #guess on x coordanites and y coordanites of centre of galaxy. use ds9 or aij
    xGuess = 574.0
    yGuess = 451.0

    #inclination and position angle of galaxy
    inclination = 27.0 
    posAngle  = 150.0
    distanceAway = 16.1*(10**6) #pc
    return (xGuess,yGuess,inclination,posAngle,distanceAway)

def annulusRadii():
    #all in arcseconds
    min    = 1.0     # inner edge of first annulus arcsec
    max    = 200.0    # outer edge of last annulus arcsec
    width       = 2.0     # annulus width 


    # Background subtraction: radius must be outside the galaxy
    skyMin    = 300.0   # inner radius of sky annulus (pixels)
    skyMax   = 500.0  # outer radius of sky annulus (pixels)
    return (min,max,width,skyMin,skyMax)


def imageCalibration():
    plate_scale    = 0.983 # pixels per arcsec
    
    zp      = 25.0 # zero point of image in instrumental mags
    return(plate_scale,zp)
