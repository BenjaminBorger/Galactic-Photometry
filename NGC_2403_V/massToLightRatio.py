import numpy as np 
from astropy.constants import G,L_sun,M_sun
from decimal import Decimal



def distModulus(m, dpc):
    # m is apparent magnitude 
    # dpc is distance in parsecs
    return (m - 5*np.log10(dpc) + 5)

def magToLum(M):
    # abs magnitude M to solar luminosities
    return pow(10, 0.4*(4.83 - M))


def massLumRelation(lum,disk_bulge):
    # lum in solar luminosities, returns mass in solar masses
    #might be higher 3.5 
    
    if disk_bulge == "disk":
        ML_ratio = 2
    else: ML_ratio = 0.5
    
    return ML_ratio * lum



def findTotalMass(v, r):
    # v is velocity in km/s
    # r in parsecs
    # Returns mass in solar masses
    
    v_mps = v * 1000  # convert to m/s
    r_m = r * 3.086e16  # convert parsecs to meters

    mass_kg = (v_mps**2 * r_m) / G.value
    
    # Convert kg to solar masses (1 solar mass = 1.989e30 kg)
    mass_solar = mass_kg / M_sun.value
    
    return mass_solar





# values in errors
#Change
#__________________________________________________________________________
#rotational velocity in km/s
RotvelocityKMS = 204
errV = 0


# distnace from centre in parsecs
radiuspc = 12000
errR = 0

#apparent magnitude of galaxy at radius r
apparentMag = 9.55259
errm = 0.01097

#___________________________________________________________________________


#distance between earth and galaxy in parsecs
distanceInParsec = 16.1*1000000
errdpc = 1.3*1000000

#absolute magnitude
absMag = distModulus(apparentMag, distanceInParsec)
errM = np.sqrt((abs(distModulus(apparentMag+errm, distanceInParsec)-distModulus(apparentMag, distanceInParsec)))**2 + (abs(distModulus(apparentMag, distanceInParsec+errdpc)-distModulus(apparentMag, distanceInParsec)))**2 )


MASS = findTotalMass(RotvelocityKMS,radiuspc)
MASSERR = 0

LUMINOSITY = magToLum(absMag)
LUMERR = abs(magToLum(absMag+errM)-magToLum(absMag))

ratioError = abs((MASS/(LUMERR+LUMINOSITY))-(MASS/(LUMINOSITY)))

print("MASS" + str('%.2E' % Decimal(MASS)))
print("LUMINOSITY: " + str('%.2E' % Decimal(LUMINOSITY))+ "+-" + str('%.2E' % Decimal(LUMERR)))
print("ratio: " + str(MASS/LUMINOSITY) + "+-" + str(ratioError))







