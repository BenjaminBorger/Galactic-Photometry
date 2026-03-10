import numpy as np

def names():
    name = "NGC 2403"
    inputFile = "Mosaic_Draco2_2022_0588_0715_NGC2403 1.fits" #string with name of mosaiced galaxy to analyse 
    outputFile = 'surface_brightness_profile2403.dat' #string with name of where surface brightness profile should go
    return(name,inputFile,outputFile)

def galaxyInfo():
    
    #guess on x coordanites and y coordanites of centre of galaxy. use ds9 or aij
    xGuess = 564.0
    yGuess = 430.0

    #inclination and position angle of galaxy tuples with (value,error)
    inclination = (63.0,0)
    posAngle  = (126,0)
    distanceAway = (3.4*(10**6),0.23*(10**6)) #pc

    return (xGuess,yGuess,inclination,posAngle,distanceAway)

def annulusRadii():
    #all in arcseconds
    min    = 1.0     # inner edge of first annulus arcsec
    max    = 450   # outer edge of last annulus arcsec
    width       = 2.0     # annulus width 


    # Background subtraction: radius must be outside the galaxy
    skyMin    = 300.0   # inner radius of sky annulus (pixels)
    skyMax   = 500.0  # outer radius of sky annulus (pixels)
    return (min,max,width,skyMin,skyMax)


def imageCalibration():
    
    #tuples with (value,error)
    plate_scale    = 0.983 # pixels per arcsec
    
    zp      = (23.675,0.003 )# zero point of image in instrumental mags and error
    return(plate_scale,zp)


def rotationCurve():
    import numpy as np
    import pandas as pd

    data = np.genfromtxt('NGC2403ROTCURVE.csv', delimiter=',', skip_header=1, dtype=None, encoding=None)

    radius_arcsec    = np.array([row[2] for row in data]) #acc in kpc
    v_rot    = np.array([row[3] for row in data])
    errVdown = np.array([row[4] for row in data])/2
    errVup = np.array([row[4] for row in data])/2
    #km/s
    '''
    radius_arcsec = np.array([4.719, 9.438, 14.55, 18.87, 23.59, 28.70, 33.42, 37.75, 42.86, 47.58,
              51.91, 57.02, 61.74, 66.46, 70.78, 75.89, 80.61, 85.33, 90.05, 93.98,
              99.10, 103.8, 108.9, 113.6, 118.7, 123.4, 128.2, 132.5, 137.2, 142.3,
              146.6, 151.0, 156.1, 160.8, 165.5, 170.2, 175.0, 180.1, 184.4, 189.5,
              194.2, 198.9, 204.1, 208.4, 212.7, 217.8, 222.5, 227.6, 232.0, 236.3,
              241.0, 246.1, 250.8, 255.2, 259.9, 265.0, 269.7, 274.4, 279.2, 284.3,
              288.2, 292.9, 298.0, 302.8, 307.5, 312.2, 317.3, 321.6, 326.7, 331.5,
              336.2, 340.5, 345.6]) # arcsec

    v_rot = np.array([9.037, 29.60, 42.14, 42.66, 50.69, 50.70, 57.23, 61.76, 64.78, 67.30,
              69.32, 72.34, 72.85, 72.87, 73.88, 73.90, 77.42, 76.93, 76.95, 79.46,
              84.49, 85.51, 85.52, 85.53, 89.56, 92.08, 90.09, 88.10, 88.61, 89.63,
              88.14, 88.15, 91.67, 89.18, 91.70, 96.22, 97.74, 98.26, 98.27, 101.7,
              102.8, 103.3, 102.8, 104.8, 105.8, 107.3, 110.4, 109.4, 111.9, 114.4,
              118.4, 120.0, 123.5, 132.0, 137.5, 140.6, 144.1, 146.1, 145.1, 141.1,
              138.1, 135.1, 134.1, 139.7, 144.7, 147.7, 147.2, 146.7, 146.8, 147.8,
              147.8, 150.3, 154.3])
    

    errVup = np.array([5.045, 10.11, 14.42, 18.74, 23.82, 28.14, 33.63, 38.32, 43.02, 47.33, 52.42, 56.74, 61.84, 66.54, 71.25, 75.56, 80.65, 84.98, 90.09, 94.78, 99.48, 103.8, 108.9, 113.6, 118.3, 123.0, 127.7, 132.8, 137.1, 141.8, 146.9, 151.2, 156.3, 161.0, 165.0, 170.4, 174.7, 179.4, 184.1, 189.2, 193.9, 198.3, 203.4, 208.1, 212.4, 217.5, 222.2, 226.5, 231.2, 235.9, 241.0, 245.7, 250.0, 255.5, 259.8, 264.5, 269.6, 273.9, 279.0, 283.4, 287.7, 292.8, 297.5, 302.2, 306.9, 311.6, 316.3, 321.0, 325.7, 330.4, 335.1, 339.4, 344.5]) # km/s, VLA channel width
    errVdown = np.array([5.503, 10.28, 15.08, 19.81, 24.55, 28.50, 33.64, 38.38, 43.09, 47.82, 52.54, 57.27, 62.39, 67.11, 71.82, 76.55, 81.27, 85.61, 90.35, 95.06, 100.1, 104.5, 109.6, 114.3, 119.1, 123.8, 128.5, 133.2, 137.9, 143.0, 147.7, 152.1, 156.8, 161.9, 166.2, 171.0, 175.7, 180.4, 185.1, 189.8, 194.6, 199.7, 204.4, 208.7, 213.5, 218.6, 223.3, 228.1, 232.4, 237.1, 241.9, 246.6, 251.7, 256.1, 260.8, 265.9, 270.7, 275.4, 279.7, 285.2, 289.5, 294.2, 298.9, 303.6, 308.4, 313.2, 318.3, 323.0, 327.3, 332.0, 336.8, 341.5, 346.2])
    '''
    errV = (errVup,errVdown)
    
    print(errV)
    #kpc
    
    errRadius = np.full_like(radius_arcsec, 15.0 * (17100 / 206265))  # ~1.24 kpc per pixel


    return v_rot,errV,radius_arcsec,errRadius


def dust():
    
    radiusDUST = [0.0, 48.0, 96.0, 144.0, 192.0]#arcsec
    errRadius = [0 for i in range(len(radiusDUST))]
    massDUST = [1.351e+07, 7.715e+07, 1.619e+08, 2.626e+08, 3.046e+08]  # M☉
    errMass = [0 for i in range(len(massDUST))]
    return radiusDUST,errRadius,massDUST,errMass