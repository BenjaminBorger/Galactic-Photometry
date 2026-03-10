import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord

def MWchecker(RA, dec):
    coord = SkyCoord(ra=RA, dec=dec, unit="deg")
    
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('parallax')
    
    result = custom_simbad.query_region(coord, radius='0.1 deg')
    
    if result is None:
        return True  # no match, assume MW star
    
    plx = result['plx_value'][0]  # in milliarcseconds
    
    if not plx or np.isnan(plx) or plx <= 0:
        return True  # no parallax, assume MW star
    
    # Convert parallax (mas) to distance (kpc): d = 1 / plx_arcsec = 1000 / plx_mas
    dist_kpc = 1000.0 / plx
    
    print(f"Parallax: {plx:.4f} mas → distance: {dist_kpc:.2f} kpc")
    
    MW_DISTANCE_THRESHOLD_KPC = 50  # MW halo extends ~50 kpc
    return dist_kpc < MW_DISTANCE_THRESHOLD_KPC