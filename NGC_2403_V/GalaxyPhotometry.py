import numpy as np # always xxx
from astropy.io import fits # allow code to read fits files
import matplotlib.pyplot as plt # plotting
from astropy.stats import SigmaClip #used for, after removing a star, filling bg back in with similar stuff
import os # allows code to search computer
import parameters # parameters file has image and galaxy info
from massToLightRatio import distModulus, magToLum, massLumRelation,findTotalMass #helper functions
from scipy.interpolate import interp1d



#input mosaic image and output file names
name, FITS_FILE, OUTPUT_FILE = parameters.names()

X_CENTER,Y_CENTER,(INCLINATION,INCLINATION_ERROR),(POSITION_ANGLE,POSITION_ANGLE_ERR),(DISTANCE,DISTANCE_ERR)  = parameters.galaxyInfo()

# Annulus radii settings
#must be floats
R_MIN_ARCSEC,R_MAX_ARCSEC,DR_ARCSEC,SKY_R_IN_PIX,SKY_R_OUT_PIX  = parameters.annulusRadii()
PLATE_SCALE,(ZERO_POINT,ZP_ERROR) = parameters.imageCalibration()


from astropy.modeling import models, fitting
from scipy.ndimage import center_of_mass
from photutils.centroids import centroid_2dg, centroid_com

def find_galaxy_centre(image, x_guess, y_guess, box_size=100):
    """
    Use centre of mass from scipy to find centre of galaxy from within a int box (box_size) centred on 
    a guess set (x_guess,y_guess) defined in parameters.py -keep general for now
   
    image is a 2D array of sci image

    returns a tuple of centre coordinates
    """
    #turns guesses into ints: needed for ease
    x0 = int(x_guess) 
    y0 = int(y_guess)

    half = box_size // 2 # makes box "radius" if you will

    #these just define the largest and smallest x y values to search within
    x_low = max(0, x0 - half)
    x_high = min(image.shape[1], x0 + half)
    y_low = max(0, y0 - half)
    y_high = min(image.shape[0], y0 + half)

    #makes a copy of the image
    search_inside_here_for_centre = image[y_low:y_high, x_low:x_high].copy()

    # Subtract local background so faint edges don't pull the centre
    bg = np.nanmedian(search_inside_here_for_centre) # finds median of little box
    
    search_inside_here_for_centre -= bg # subtracts median so all is well and good

    search_inside_here_for_centre = np.clip(search_inside_here_for_centre, 0, None)  # zero out negatives
    cy, cx = center_of_mass(np.nan_to_num(search_inside_here_for_centre)) # actually finds centre of mass
    results = (cx + x_low, cy + y_low) # creates tuple with xy coordanites for centre
    return results

def deproject_galaxy(x, y, x_center, y_center, inclination, position_angle):
    i  = np.radians(inclination)
    pa = np.radians(90 - position_angle)

    dx = x - x_center
    dy = y - y_center

    # Rotate so major axis aligns with x-axis
    x_rot =  dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)

    # Stretch minor axis to deproject
    x_dep = x_rot
    y_dep = y_rot / np.cos(i)

    return np.sqrt(x_dep**2 + y_dep**2)


def remove_large_star(image, star_x, star_y, star_radius, 
                      interpolate=True, feather=5):
    """
    use star detector script for parameters
    """
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((xx - star_x)**2 + (yy - star_y)**2)

    cleaned = image.copy().astype(float)

    if not interpolate:
        cleaned[dist <= star_radius] = np.nan
        return cleaned

    # Sample background from annulus just outside the star
    sky_in  = star_radius * 1.1
    sky_out = star_radius * 1.5
    sky_annulus = (dist >= sky_in) & (dist <= sky_out)
    sky_pixels  = image[sky_annulus]
    sky_pixels  = sky_pixels[np.isfinite(sky_pixels)]

    if len(sky_pixels) == 0:
        print("NaN")
        cleaned[dist <= star_radius] = np.nan
        return cleaned

    sky_mean = np.median(sky_pixels)
    sky_std  = np.std(sky_pixels)

    mask = dist <= star_radius
    n_fill = np.sum(mask)
    fill_values = np.random.normal(sky_mean, sky_std, size=n_fill)
    cleaned[mask] = fill_values
    if feather > 0:
        for f in range(1, feather + 1):
            blend_mask = (dist > star_radius - feather + f - 1) & \
                         (dist <= star_radius - feather + f)
            if not np.any(blend_mask):
                continue
            weight = f / (feather + 1)  # 0 = all fill, 1 = all original
            n_blend = np.sum(blend_mask)
            blend_fill = np.random.normal(sky_mean, sky_std, size=n_blend)
            cleaned[blend_mask] = (1 - weight) * blend_fill + \
                                   weight * image[blend_mask]

    print(f"Removed star at ({star_x:.1f}, {star_y:.1f}) "
          f"with radius {star_radius:.1f} px. "
          f"Sky fill: {sky_mean:.2f} ± {sky_std:.2f} counts.")

    return cleaned



def estimate_sky_background(image, x_cen, y_cen, r_in, r_out):
    """Median sky background from a circular annulus far from the galaxy."""
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - x_cen)**2 + (yy - y_cen)**2)
    sky_mask = (r >= r_in) & (r <= r_out)
    sky_pixels = image[sky_mask]
    sky_pixels = sky_pixels[np.isfinite(sky_pixels)]
    sky_median = np.median(sky_pixels)
    sky_std    = np.std(sky_pixels)
    return sky_median, sky_std


def measure_annulus_profile(image, x_cen, y_cen,
                            r_min_arcsec, r_max_arcsec, dr_arcsec,
                            plate_scale, inclination, position_angle,
                            zero_point, sky_bg):
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    r_dep = deproject_galaxy(xx, yy, x_cen, y_cen, inclination, position_angle)

    img_sub = image - sky_bg

    radii_arcsec = np.arange(r_min_arcsec, r_max_arcsec, dr_arcsec)
    results = []

    for r_inner in radii_arcsec:
        r_outer = r_inner + dr_arcsec
        r_mid   = (r_inner + r_outer) / 2.0

        r_in_pix  = r_inner * plate_scale
        r_out_pix = r_outer * plate_scale

        mask = (r_dep >= r_in_pix) & (r_dep < r_out_pix)
        pixels = img_sub[mask]
        pixels = pixels[np.isfinite(pixels)]

        if len(pixels) == 0:
            continue

        n_pix       = len(pixels)
        total_flux  = np.sum(pixels)
        mean_flux   = np.mean(pixels)
        median_flux = np.median(pixels)
        flux_err    = np.std(pixels) / np.sqrt(n_pix)

        if total_flux > 0:
            # Convert directly from total flux to luminosity
            apparent_mag = zero_point - 2.5 * np.log10(total_flux)
            absolute_mag = distModulus(apparent_mag, DISTANCE)
            Lum_mag       = magToLum(absolute_mag)  # now luminosity in L_sun

            # Error propagation: σ_mag = 1.0857 * σ_flux/flux
            # then propagate through distModulus and magToLum via finite difference
            flux_shifted     = total_flux + flux_err
            apparent_shifted = zero_point - 2.5 * np.log10(flux_shifted)
            absolute_shifted = distModulus(apparent_shifted, DISTANCE)
            lum_shifted      = magToLum(absolute_shifted)
            Lum_err           = abs(Lum_mag - lum_shifted)  # luminosity error

            # also propagate ZP_ERROR
            apparent_zp  = (zero_point + ZP_ERROR) - 2.5 * np.log10(total_flux)
            absolute_zp  = distModulus(apparent_zp, DISTANCE)
            lum_zp       = magToLum(absolute_zp)
            zp_err_lum   = abs(Lum_mag - lum_zp)

            # propagate distance error
            apparent_dist = zero_point - 2.5 * np.log10(total_flux)
            absolute_dist = distModulus(apparent_dist, DISTANCE + DISTANCE_ERR)
            lum_dist      = magToLum(absolute_dist)
            dist_err_lum  = abs(Lum_mag - lum_dist)

            # combine in quadrature
            Lum_err = np.sqrt(Lum_err**2 + zp_err_lum**2 + dist_err_lum**2)
        else:
            Lum_mag = np.nan
            Lum_err = np.nan

        results.append({
            'r_arcsec'    : r_mid,
            'r_in_arcsec' : r_inner,
            'r_out_arcsec': r_outer,
            'n_pixels'    : n_pix,
            'total_flux'  : total_flux,
            'mean_flux'   : mean_flux,
            'median_flux' : median_flux,
            'flux_err'    : flux_err,
            'Lum_mag'      : Lum_mag,  
            'Lum_err'      : Lum_err,  
        })

    return results
def save_profile(results, filename, sky_bg, sky_std, x_cen, y_cen,
                 inclination, position_angle, zero_point):
    """Save the radial profile to a .dat file with a descriptive header."""
    with open(filename, 'w') as f:
        f.write("# Radial Surface Brightness Profile\n")
        f.write(f"# Galaxy centre (x, y) pixels : {x_cen:.2f}, {y_cen:.2f}\n")
        f.write(f"# Inclination                  : {inclination:.1f} deg\n")
        f.write(f"# Position angle               : {position_angle:.1f} deg\n")
        f.write(f"# Plate scale                  : {PLATE_SCALE:.4f} pix/arcsec\n")
        f.write(f"# Sky background (median)      : {sky_bg:.4f} counts\n")
        f.write(f"# Sky background (std)         : {sky_std:.4f} counts\n")
        f.write(f"# Photometric zero-point       : {zero_point:.2f} mag\n")
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("#  r_mid_arcsec  r_in_arcsec  r_out_arcsec  n_pixels  "
                "total_flux  mean_flux  median_flux  flux_err  Lum_mag  Lum_err\n")

        for row in results:
            f.write(
                f"{row['r_arcsec']:10.4f}  "
                f"{row['r_in_arcsec']:10.4f}  "
                f"{row['r_out_arcsec']:10.4f}  "
                f"{row['n_pixels']:8d}  "
                f"{row['total_flux']:14.4f}  "
                f"{row['mean_flux']:12.4f}  "
                f"{row['median_flux']:12.4f}  "
                f"{row['flux_err']:12.4f}  "
                f"{row['Lum_mag']:8.4f}  "
                f"{row['Lum_err']:8.4f}\n"
            )
    print(f"Profile saved to: {filename}")



#need errors for values of cumulative luminous mass, cumulateive real mass, cumulative dust mass, mass to light, unaccounted matter
def findLumFromLum(zp,sb,N,dist):
        flux_per_pix = 10**((zp - sb) / 2.5)
        total_flux   = flux_per_pix * N
        valid = np.isfinite(sb) & (total_flux > 0)

        apparent_mag = zp - 2.5 * np.log10(total_flux[valid])
        absolute_mag = distModulus(apparent_mag, dist)
        luminosity = magToLum(absolute_mag)
        return luminosity


def getErrorsFUNCTIONAL(function, params, errors):
    import numpy as np
    SummatedErrSq = 0
    for i in range(len(params)):
        params_shifted = list(params)  # shallow copy as list
        params_shifted[i] = params[i] + errors[i]
        diff = function(*params) - function(*params_shifted)
        SummatedErrSq += diff**2

    return np.sqrt(SummatedErrSq)

def CUMSUMERRORS(arr):
    arr = np.array(arr)
    return np.sqrt(np.cumsum(arr**2))
    
def plot_profile(results, output_prefix='profile'):
    """Quick diagnostic plot of the surface brightness profile."""
    r_arcsec_lum   = np.array([d['r_arcsec'] for d in results])
    Lum  = np.array([d['Lum_mag']   for d in results])
    err = np.array([d['Lum_err']   for d in results])

    valid = np.isfinite(Lum)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Surface brightness vs radius
    ax = axes[0]
    ax.errorbar(r_arcsec_lum[valid], Lum[valid], yerr=err[valid], fmt='o-', ms=4, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('Deprojected radius (arcsec)')
    ax.set_ylabel(f'Surface brightness (mag/pix, ZP={ZERO_POINT})')
    ax.set_title('Surface Brightness Profile')

    # Mean flux vs radius (linear)
    flux = np.array([d['mean_flux'] for d in results])
    axes[1].semilogy(r_arcsec_lum, flux, 'o-', ms=4)
    axes[1].set_xlabel('Deprojected radius (arcsec)')
    axes[1].set_ylabel('Mean flux (counts/pixel)')
    axes[1].set_title('Flux Profile (log scale)')

    plt.tight_layout()
    plot_file = output_prefix + '.png'
    plt.savefig(plot_file, dpi=150)
    print(f"Diagnostic plot saved to: {plot_file}")
    plt.show()

def removeStars(keepID, image):
    det_stars = "detected_stars.dat"
    cleaned = image.copy()

    with open(det_stars, "r") as f:
        lines = f.readlines()

    if not lines:
        print("detected_stars.dat is empty")
        return cleaned

    # First line is the header (no # prefix)
    header = lines[0].strip().lstrip('#').split()
    print(f"Parsed header: {header}")

    for line in lines[1:]:  # skip header line
        if line.strip() == '' or line.strip().startswith('#'):
            continue
        columns = line.split()
        try:
            ID   = int(float(columns[header.index('id')]))
            if ID in keepID:
                continue
            x    = float(columns[header.index('xcentroid')])
            y    = float(columns[header.index('ycentroid')])
            fwhm = float(columns[header.index('fwhm')])
        except (ValueError, IndexError) as e:
            print(f"Skipping line: {e}")
            continue

        peak  = float(columns[header.index('peak')])
        star_radius = max(3.0 * fwhm, 2.5 * fwhm * np.log10(max(peak, 10)))

        print(f"Removing star ID={ID} at ({x:.1f},{y:.1f}) with radius {star_radius:.1f}px")
        cleaned = remove_large_star(cleaned, star_x=x, star_y=y, star_radius=star_radius)

    return cleaned


def plot_galaxy_with_profile(image, results, x_cen, y_cen,
                              output_file='galaxy_profile_overlay.png'):
    
    
    from scipy.interpolate import interp1d
    fig = plt.figure(figsize=(14, 6))
   
    kpc_per_arcsec = (DISTANCE / 1000) * (np.pi / (180 * 3600))
    
    from fitsHeaderGetter import pixelCoordinatesToRADEC

    # image of galaxy
    ax_img = fig.add_subplot(231)
    vmin, vmax = np.nanpercentile(image, [1, 99])
    ax_img.imshow(image, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)

    from matplotlib.patches import Ellipse
    r_arcsec = np.array([d['r_out_arcsec'] for d in results])
    for r in r_arcsec[::9]:
        a = r * PLATE_SCALE                          # semi-major axis in pixels
        b = a * np.cos(np.radians(INCLINATION))      # semi-minor axis
        ellipse = Ellipse(
            xy=(x_cen, y_cen),
            width=2 * a,
            height=2 * b,
            angle= 90+POSITION_ANGLE,                    # degrees, matplotlib uses CCW from x-axis
            color='white', fill=False, linewidth=0.5, alpha=0.7
        )
        ax_img.add_patch(ellipse)

    ax_img.plot(x_cen, y_cen, '+', color='cyan', ms=12, mew=1.5, label='Centre')
    ax_img.legend(loc='upper right', fontsize=8)
    ax_img.set_title(f'{name} (Stars Removed)')

    x_ticks_px = ax_img.get_xticks()
    y_ticks_px = ax_img.get_yticks()

    x_tick_labels = [f'{pixelCoordinatesToRADEC(px, y_cen)[0]:.4f}' for px in x_ticks_px]
    y_tick_labels = [f'{pixelCoordinatesToRADEC(x_cen, py)[1]:.4f}' for py in y_ticks_px]

    ax_img.set_xticks(x_ticks_px)
    ax_img.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=7)
    ax_img.set_yticks(y_ticks_px)
    ax_img.set_yticklabels(y_tick_labels, fontsize=7)

    ax_img.set_xlabel('RA (degrees)')
    ax_img.set_ylabel('Dec (degrees)')

    # surface brightness profile
    ax_prof = fig.add_subplot(232)
    r_arcsec_lum   = np.array([d['r_arcsec'] for d in results]) # in arcsec 
    Lum  = np.array([d['Lum_mag']   for d in results])
    err = np.array([d['Lum_err']   for d in results])
    N = np.array([d['n_pixels']   for d in results]) #  num of pixels - times by magnitude?
    valid = np.isfinite(Lum)
    
    
    #pair with if using surface brightness
    ''''
    #get luminosity from surface brightness
    luminosity = findLumFromsb(ZERO_POINT,sb,N,DISTANCE)
    cumulative_luminosity = np.cumsum(luminosity)
    #errors. will cumsum lum so need to do cumsum error
    lumErr = getErrorsFUNCTIONAL(findLumFromsb, [ZERO_POINT,sb[valid],N[valid],DISTANCE], [0.0002,err[valid],0,DISTANCE_ERR] )
    cumsum_error_lum = CUMSUMERRORS(lumErr)
    '''
    luminosity = Lum[valid] # cant be borthered to change from Lum
    cumulative_luminosity = np.cumsum(luminosity)
    lumErr = err[valid]
    cumsum_error_lum = CUMSUMERRORS(lumErr)



    ax_prof.errorbar((r_arcsec_lum[valid]*kpc_per_arcsec)[::5], cumulative_luminosity[::5] , yerr=cumsum_error_lum[::5],
                     fmt='o-', ms=4, capsize=3, color='black', ecolor='black')
    ax_prof.set_xlabel('Deprojected radius (kPc)')
    ax_prof.set_ylabel(f'Luminosity with ZP={ZERO_POINT})')
    ax_prof.set_title('Luminosity against Radius')

    #ROTATION CURVE DATA
    v_rot, (v_rot_errUP,v_rot_errDown), radius_kpc, r_kpc_err = parameters.rotationCurve()
    
    v_rot_err = [v_rot_errDown,v_rot_errUP]
    print(v_rot_err)
    #radius to parsecs
    # parsecs, needed for findTotalMass
    radius_noninterp_pc = radius_kpc * 1000
    radius_arcsec = (radius_noninterp_pc / DISTANCE) * (180 / np.pi) * 3600
    #radius_kpc = (radius_arcsec*np.pi*DISTANCE)/(180*3600*1000)
    
    
    realMasses = np.array([findTotalMass(v_rot[k], radius_noninterp_pc[k]) for k in range(len(radius_noninterp_pc))])
    

    bulgeRangeARCSEC = PLATE_SCALE*20
    enclosedMass = []
    enclosedMassErr = []
    for i in range(len(r_arcsec_lum[valid])):
        if r_arcsec_lum[valid][i] <= bulgeRangeARCSEC:
          enclosedMass.append(massLumRelation(luminosity[i], "bulge"))
          enclosedMassErr.append(abs(massLumRelation(luminosity[i], "bulge")-massLumRelation(luminosity[i]+lumErr[i], "bulge")))
        else:
          enclosedMass.append(massLumRelation(luminosity[i], "disk"))
          enclosedMassErr.append(abs(massLumRelation(luminosity[i], "disk")-massLumRelation(luminosity[i]+lumErr[i], "disk")))

    cumulativeStellar = np.cumsum(enclosedMass)
    stellarMass_Err = CUMSUMERRORS(enclosedMassErr)

    def getErrorsOneSided(function, params, errors):
        SummatedErrSq = 0
        for i in range(len(params)):
            params_shifted = list(params)
            params_shifted[i] = params[i] + errors[i]
            diff = function(*params) - function(*params_shifted)
            SummatedErrSq += diff**2
        return np.sqrt(SummatedErrSq)

    # Upper mass error (v goes up)
    realMassErrors_up = np.array([
        getErrorsOneSided(findTotalMass, [v_rot[k], radius_noninterp_pc[k]], 
                                        [v_rot_errUP[k], r_kpc_err[k]])
        for k in range(len(radius_noninterp_pc))
    ])

    realMassErrors_down = np.array([
        getErrorsOneSided(findTotalMass, [v_rot[k], radius_noninterp_pc[k]], 
                                        [-v_rot_errDown[k], -r_kpc_err[k]])
        for k in range(len(radius_noninterp_pc))
    ])

    #do this for the ratio too 
    #    
    minimum_radius = np.min([r_arcsec_lum[-1],radius_arcsec[-1]])
    print("min radius")
    print(minimum_radius*kpc_per_arcsec)

    def massLumRatio(cumsumLum,cumsumMass):
        return cumsumMass/cumsumLum
    
    
    ax_prof = fig.add_subplot(233)
    
    #inerpolate dynamical mass and errors

    dynamicalMass_interp = interp1d(radius_arcsec, realMasses, bounds_error=False, fill_value='extrapolate')
    interpolatedMass = dynamicalMass_interp(r_arcsec_lum[valid])
    
    interpMassErrors_up   = interp1d(radius_arcsec, realMassErrors_up,   bounds_error=False, fill_value='extrapolate')
    interpMassErrors_down = interp1d(radius_arcsec, realMassErrors_down, bounds_error=False, fill_value='extrapolate')
    interpolatedMassErrors_up   = interpMassErrors_up(r_arcsec_lum[valid])
    interpolatedMassErrors_down = interpMassErrors_down(r_arcsec_lum[valid])

    lumInterp = interp1d(r_arcsec_lum[valid],cumulative_luminosity,bounds_error=False, fill_value='extrapolate')
    interpolatedLuminosity = lumInterp(r_arcsec_lum[valid])

    lum_ERR_Interp = interp1d(r_arcsec_lum[valid],cumsum_error_lum,bounds_error=False, fill_value='extrapolate')
    interpolatedLuminosity_ERR = lum_ERR_Interp(r_arcsec_lum[valid])

    print(type(lumErr), np.shape(lumErr))

    massLumratioErr_up = getErrorsFUNCTIONAL(massLumRatio,
                            [interpolatedLuminosity, interpolatedMass],
                            [interpolatedLuminosity_ERR, interpolatedMassErrors_up])

    massLumratioErr_down = getErrorsFUNCTIONAL(massLumRatio,
                            [interpolatedLuminosity, interpolatedMass],
                            [interpolatedLuminosity_ERR, interpolatedMassErrors_down])  
      
    ax_prof.errorbar((r_arcsec_lum[valid]*kpc_per_arcsec)[::3],
                    massLumRatio(interpolatedLuminosity, interpolatedMass)[::3],
                    yerr=[massLumratioErr_down[::3], massLumratioErr_up[::3]])    
    ax_prof.set_xlim(left=0)   # x axis starts at 0
    ax_prof.set_ylim(bottom=0) # y axis starts at 0

# or set both ends
    ax_prof.set_xlim(left = 0)
    #cumulative luminosity since its cumulative mass (fucking idiot)
    
    #luminosity corresponds to r_arcsec_lum
    ax_prof.plot(r_arcsec_lum[valid]*kpc_per_arcsec,(interpolatedMass/cumulative_luminosity), color = 'black')
    
    ax_prof.set_xlabel('Deprojected radius (kPc)')
    ax_prof.set_ylabel(r'$\Upsilon_\odot$')
    ax_prof.set_title('Mass To Light Ratio as FuncRad')

    #confounding factor of dust
    radiusDUST, radiusDUST_err, massDUST, massDUST_err = parameters.dust()

    # Interpolate cumulative stellar mass onto the dynamical mass radii
    stellar_interp = interp1d(r_arcsec_lum[valid], cumulativeStellar, bounds_error=False, fill_value='extrapolate')
    dust_interp = interp1d(radiusDUST, massDUST, bounds_error=False, fill_value='extrapolate')
    stellar_at_dyn = stellar_interp(radius_arcsec)
    dust_at_dyn = dust_interp(radius_arcsec)

    # interpolate stellar and dust errors onto the dynamical mass radii
    stellarMass_Err_interp = interp1d(r_arcsec_lum[valid], stellarMass_Err, 
                                    bounds_error=False, fill_value='extrapolate')
    stellar_err_at_dyn = stellarMass_Err_interp(radius_arcsec)

    dust_err_interp = interp1d(radiusDUST, massDUST_err, 
                                bounds_error=False, fill_value='extrapolate')
    dust_err_at_dyn = dust_err_interp(radius_arcsec)

    def unseenMassFunc(realMass, stellar, dust):
        return realMass - stellar - dust

    unseenMass_err_up = getErrorsFUNCTIONAL(unseenMassFunc,
                            [realMasses, stellar_at_dyn, dust_at_dyn],
                            [realMassErrors_up, stellar_err_at_dyn, dust_err_at_dyn])

    unseenMass_err_down = getErrorsFUNCTIONAL(unseenMassFunc,
                            [realMasses, stellar_at_dyn, dust_at_dyn],
                            [-realMassErrors_down, stellar_err_at_dyn, dust_err_at_dyn])

    ax_prof = fig.add_subplot(2,3,4)
    offset = 0.05
    vals = unseenMassFunc(realMasses, stellar_at_dyn, dust_at_dyn)
    lower_err = np.minimum(unseenMass_err_down, np.maximum(vals, 0))
    ax_prof.set_xlabel('Deprojected radius (kPc)')
    ax_prof.set_ylabel('Enclosed Unseen Mass (M☉)')
    ax_prof.set_title('Cumulative Unseen Mass against Radius')

    

    #find index at which these plots exceed minimum radius
    overallMaxRadius = min(r_arcsec_lum[valid][-1], radius_arcsec[-1]) * kpc_per_arcsec
    overallMinRadius = max(r_arcsec_lum[valid][0], radius_arcsec[0]) * kpc_per_arcsec
    # use r_arcsec_lum[valid] for stellar since cumulativeStellar has len sum(valid)
    r_arcsec_lum_valid = r_arcsec_lum[valid]
    
    cut_dynHi = np.searchsorted(radius_kpc, overallMaxRadius)
    cut_dynLo = np.searchsorted(radius_kpc, overallMinRadius)
    
    cut_stellarHi = np.searchsorted(r_arcsec_lum_valid * kpc_per_arcsec, overallMaxRadius)
    cut_stellarLo = np.searchsorted(r_arcsec_lum_valid * kpc_per_arcsec, overallMinRadius)
    
    cut_dustHi = np.searchsorted(np.array(radiusDUST)*kpc_per_arcsec, overallMaxRadius)
    cut_dustLo = np.searchsorted(np.array(radiusDUST)*kpc_per_arcsec, overallMinRadius)


    ax_prof.errorbar(radius_arcsec*kpc_per_arcsec + offset, 
                    vals,
                    yerr=[lower_err, unseenMass_err_up],
                    color='black', label='matter which does not interact with V Band',
                    fmt='o', ms=2, capsize=2)
    ax_prof.plot(radius_arcsec*kpc_per_arcsec, 
                    vals,
                    color='black')
    

    ax_prof.set_xlim(right = overallMaxRadius)
    ax_prof.set_ylim(top = vals[cut_dynHi])



    ax_prof = fig.add_subplot(2,3,(5,6))
    ax_prof.errorbar(r_arcsec_lum_valid[cut_stellarLo:cut_stellarHi]*kpc_per_arcsec,
                     cumulativeStellar[cut_stellarLo:cut_stellarHi],
                     yerr=stellarMass_Err[cut_stellarLo:cut_stellarHi],
                     color='red', label='Luminous mass', fmt='o', ms=2, capsize=2)


    ax_prof.errorbar(np.array(radiusDUST[cut_dustLo:cut_dustHi])*kpc_per_arcsec,
                     massDUST[cut_dustLo:cut_dustHi],
                     yerr=radiusDUST_err[cut_dustLo:cut_dustHi],
                     color='grey', label='Mass from dust', fmt='o', ms=2, capsize=2)

    ax_prof.errorbar(np.array(radius_kpc[cut_dynLo:cut_dynHi]),
                     realMasses[cut_dynLo:cut_dynHi],
                     yerr=[realMassErrors_down[cut_dynLo:cut_dynHi], realMassErrors_up[cut_dynLo:cut_dynHi]],
                     color='blue', label='Dynamical (total) mass', fmt='o', ms=2, capsize=2)

    ax_prof.set_xlabel('Deprojected radius (kPc)')
    ax_prof.set_ylabel('Enclosed Mass (M☉)')
    ax_prof.set_title('Cumulative Dynamical Mass')
    ax_prof.legend()
    #ax_prof.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Overlay plot saved to: {output_file}")
    plt.show()
    #print(f"{unseenMass[-1]/realMasses[-1]} percent of this galaxy does not interact with light")



# main function
if __name__ == '__main__':

    hdul  = fits.open(FITS_FILE)
    image = hdul[0].data.astype(float)
    header = hdul[0].header
    hdul.close()

    X_CENTER, Y_CENTER = find_galaxy_centre(image, X_CENTER, Y_CENTER, box_size=150)


    print(f"Image shape : {image.shape}")
    print(f"Galaxy centre: ({X_CENTER}, {Y_CENTER}) pixels")
    
    
    #add in the argument the ID if it flags the galaxy in any counts
    #__________________________________________________________________
    # want to confirm keeps are in fact stars IN milky way
    
    keeps = []
    image = removeStars(keeps,image)
    #__________________________________________________________________
    
    # sky background
    sky_bg, sky_std = estimate_sky_background(
        image, X_CENTER, Y_CENTER, SKY_R_IN_PIX, SKY_R_OUT_PIX
    )
    print(f"Sky background: {sky_bg:.4f} ± {sky_std:.4f} counts")

    # annular profile 
    results = measure_annulus_profile(
        image, X_CENTER, Y_CENTER,
        R_MIN_ARCSEC, R_MAX_ARCSEC, DR_ARCSEC,
        PLATE_SCALE, INCLINATION, POSITION_ANGLE,
        ZERO_POINT, sky_bg
    )

    print(f"\nMeasured {len(results)} annuli from {R_MIN_ARCSEC}\" to {R_MAX_ARCSEC}\"")
    print(f"{'r_mid\":':>10} {'Lum (mag/pix)':>14} {'±':>4} {'N_pix':>8}")
    print("-" * 42)
    for row in results:
        print(f"{row['r_arcsec']:10.2f} {row['Lum_mag']:14.4f} {row['Lum_err']:6.4f} {row['n_pixels']:8d}")

    # Save to .dat
    save_profile(results, OUTPUT_FILE, sky_bg, sky_std,
                 X_CENTER, Y_CENTER, INCLINATION, POSITION_ANGLE, ZERO_POINT)

    # Plot
    #plot_profile(results, output_prefix=os.path.splitext(OUTPUT_FILE)[0])
    plot_galaxy_with_profile(
    image,      
    results,
    X_CENTER, Y_CENTER,
    output_file='galaxy_profile_overlay.png'
)