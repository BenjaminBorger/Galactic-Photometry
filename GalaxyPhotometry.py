import numpy as np # always xxx
from astropy.io import fits # allow code to read fits files
import matplotlib.pyplot as plt # plotting
from astropy.stats import SigmaClip #used for, after removing a star, filling bg back in with similar stuff
import os # allows code to search computer
import parameters # parameters file has image and galaxy info
from massToLightRatio import distModulus, massToLum, massLumRelation,findTotalMass

#input mosaic image and output file names
FITS_FILE, OUTPUT_FILE = parameters.names()

X_CENTER,Y_CENTER,INCLINATION,POSITION_ANGLE,DISTANCE  = parameters.galaxyInfo()

# Annulus radii settings
#must be floats
R_MIN_ARCSEC,R_MAX_ARCSEC,DR_ARCSEC,SKY_R_IN_PIX,SKY_R_OUT_PIX  = parameters.annulusRadii()
PLATE_SCALE,ZERO_POINT = parameters.imageCalibration()


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
    """
    Deproject galaxy coordinates to face-on view.

    Parameters
    ----------
    x, y            : array-like  — pixel coordinates
    x_center/y_center: float      — galaxy centre
    inclination     : float       — degrees (0=face-on, 90=edge-on)
    position_angle  : float       — degrees, CCW from North (+Y axis)

    Returns
    -------
    r_dep : deprojected radial distance in pixels
    """
    i  = np.radians(inclination)
    pa = np.radians(position_angle)

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
        print("Warning: no sky pixels found in annulus, filling with NaN.")
        cleaned[dist <= star_radius] = np.nan
        return cleaned

    sky_mean = np.median(sky_pixels)
    sky_std  = np.std(sky_pixels)

    # Fill masked region with sky + noise to look natural
    mask = dist <= star_radius
    n_fill = np.sum(mask)
    fill_values = np.random.normal(sky_mean, sky_std, size=n_fill)
    cleaned[mask] = fill_values

    # Feather the edge: blend original and fill over `feather` pixels
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


# Call once per large star you want to remove



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
    """
    Measure surface brightness in deprojected circular annuli.

    Returns list of dicts with photometric results per annulus.
    """
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Deprojected radius map (pixels)
    
    r_dep = deproject_galaxy(xx, yy, x_cen, y_cen, inclination, position_angle)

    # Sky-subtracted image
    img_sub = image - sky_bg

    radii_arcsec = np.arange(r_min_arcsec, r_max_arcsec, dr_arcsec)
    results = []

    for r_inner in radii_arcsec:
        r_outer = r_inner + dr_arcsec
        r_mid   = (r_inner + r_outer) / 2.0

        # Convert radii to pixels for the mask
        r_in_pix  = r_inner * plate_scale
        r_out_pix = r_outer * plate_scale

        mask = (r_dep >= r_in_pix) & (r_dep < r_out_pix)
        pixels = img_sub[mask]
        pixels = pixels[np.isfinite(pixels)]

        if len(pixels) == 0:
            continue

        n_pix        = len(pixels)
        total_flux   = np.sum(pixels)
        mean_flux    = np.mean(pixels)
        median_flux  = np.median(pixels)
        flux_err     = np.std(pixels) / np.sqrt(n_pix)  # standard error on mean

        # Surface brightness: magnitude per pixel (mean flux in annulus)
        if mean_flux > 0:
            sb_mag = zero_point - 2.5 * np.log10(mean_flux)
            # Propagate error: δm = 1.0857 * δF/F
            sb_err = 1.0857 * (flux_err / mean_flux)
        else:
            sb_mag = np.nan
            sb_err = np.nan

        results.append({
            'r_arcsec'   : r_mid,
            'r_in_arcsec': r_inner,
            'r_out_arcsec': r_outer,
            'n_pixels'   : n_pix,
            'total_flux' : total_flux,
            'mean_flux'  : mean_flux,
            'median_flux': median_flux,
            'flux_err'   : flux_err,
            'sb_mag'     : sb_mag,
            'sb_err'     : sb_err,
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
                "total_flux  mean_flux  median_flux  flux_err  sb_mag  sb_err\n")

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
                f"{row['sb_mag']:8.4f}  "
                f"{row['sb_err']:8.4f}\n"
            )
    print(f"Profile saved to: {filename}")


def plot_profile(results, output_prefix='profile'):
    """Quick diagnostic plot of the surface brightness profile."""
    r   = np.array([d['r_arcsec'] for d in results])
    sb  = np.array([d['sb_mag']   for d in results])
    err = np.array([d['sb_err']   for d in results])

    valid = np.isfinite(sb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Surface brightness vs radius
    ax = axes[0]
    ax.errorbar(r[valid], sb[valid], yerr=err[valid], fmt='o-', ms=4, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('Deprojected radius (arcsec)')
    ax.set_ylabel(f'Surface brightness (mag/pix, ZP={ZERO_POINT})')
    ax.set_title('Surface Brightness Profile')
    ax.grid(True, alpha=0.3)

    # Mean flux vs radius (linear)
    flux = np.array([d['mean_flux'] for d in results])
    axes[1].semilogy(r, flux, 'o-', ms=4)
    axes[1].set_xlabel('Deprojected radius (arcsec)')
    axes[1].set_ylabel('Mean flux (counts/pixel)')
    axes[1].set_title('Flux Profile (log scale)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_prefix + '.png'
    plt.savefig(plot_file, dpi=150)
    print(f"Diagnostic plot saved to: {plot_file}")
    plt.show()

def removeStars(keepID, image):
    det_stars = "detected_stars.dat"
    cleaned = image.copy()

    with open(det_stars, "r") as f:
        for row in f:
            columns = row.split()
            try:
                ID = int(columns[0])
                if ID in keepID:
                    continue
                x = float(columns[1])   # ← inside try
                y = float(columns[2])   # ← inside try
            except (ValueError, IndexError):
                continue

            cleaned = remove_large_star(cleaned, star_x=x, star_y=y, star_radius=25.0)

    return cleaned
def plot_galaxy_with_profile(image, results, x_cen, y_cen,
                              output_file='galaxy_profile_overlay.png'):
    """
    Plot the galaxy image with the surface brightness profile overlaid.
    
    Parameters
    ----------
    image       : 2D array — the science image
    results     : list of dicts — output from measure_annulus_profile
    x_cen/y_cen : float — galaxy centre in pixels
    output_file : str — output filename
    """
    fig = plt.figure(figsize=(14, 6))

    # image of galaxy
    ax_img = fig.add_subplot(231)
    vmin, vmax = np.nanpercentile(image, [1, 99])
    ax_img.imshow(image, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)

    # annuli on galaxy image
    r_arcsec = np.array([d['r_out_arcsec'] for d in results])
    for r in r_arcsec[::3]:  # every 3rd annulus to avoid clutter
        circle = plt.Circle((x_cen, y_cen), r * PLATE_SCALE,
                             color='white', fill=False, linewidth=0.5, alpha=0.4)
        ax_img.add_patch(circle)

    ax_img.plot(x_cen, y_cen, '+', color='cyan', ms=12, mew=1.5, label='Centre')
    ax_img.legend(loc='upper right', fontsize=8)
    ax_img.set_title('Galaxy Image')
    ax_img.set_xlabel('x (pixels)')
    ax_img.set_ylabel('y (pixels)')

    # surface brightness profile
    ax_prof = fig.add_subplot(232)
    r   = np.array([d['r_arcsec'] for d in results]) # in arcsec 
    sb  = np.array([d['sb_mag']   for d in results])
    err = np.array([d['sb_err']   for d in results])
    N = np.array([d['n_pixels']   for d in results]) #  num of pixels - times by magnitude?
    valid = np.isfinite(sb)
    

    ax_prof.errorbar(r[valid], sb[valid] , yerr=err[valid],
                     fmt='o-', ms=4, capsize=3, color='black', ecolor='black')
    ax_prof.invert_yaxis()
    ax_prof.set_xlabel('Deprojected radius (arcsec)')
    ax_prof.set_ylabel(f'Surface brightness (mag/pix, ZP={ZERO_POINT})')
    ax_prof.set_title('Surface Brightness Profile')
    ax_prof.grid(True, alpha=0.3)
    
    ###fix mass issue on scale
    

    flux_per_pix = 10**((ZERO_POINT - sb) / 2.5)
    total_flux   = flux_per_pix * N

    valid = np.isfinite(sb) & (total_flux > 0)

    apparent_mag = ZERO_POINT - 2.5 * np.log10(total_flux[valid])
    absolute_mag = distModulus(apparent_mag, DISTANCE)
    luminosity = massToLum(absolute_mag)
    
    ax_prof = fig.add_subplot(233)
    enclosedMass = massLumRelation(luminosity)
    cumulativeStellar = np.cumsum(enclosedMass)

    ax_prof.plot(r[valid], cumulativeStellar, color='black', label='Stellar mass')
    ax_prof.set_xlabel('Deprojected radius (arcsec)')
    ax_prof.set_ylabel('Cumulative Mass (M☉)')
    ax_prof.set_title('Cumulative Stellar Mass')
    ax_prof.grid(True, alpha=0.3)


    radius_kpc = np.array([0.02, 0.41, 0.83, 1.24, 1.66, 2.07, 2.49, 2.90, 3.31, 3.73,
      4.14, 4.56, 4.97, 5.39, 5.80, 6.21, 6.63, 7.04, 7.46, 7.87,
      8.29, 8.70, 9.11, 9.53, 9.94])

    radius = radius_kpc * 1000  # parsecs, still needed for findTotalMass

    radius_arcsec = (radius / DISTANCE) * (180 / np.pi) * 3600

    v_rot = np.array([25.15, 62.78, 86.50, 105.31, 124.12, 137.65, 151.55, 163.00,
         171.17, 179.34, 188.33, 198.15, 205.51, 212.05, 218.59, 221.88,
         227.60, 230.86, 234.94, 241.66, 245.71, 249.74, 253.77, 256.48, 260.51])
    
    ax_prof = fig.add_subplot(2,3,(4,6))
    realMasses = np.array([findTotalMass(v_rot[k], radius[k]) for k in range(len(radius))])


    #confounding factor of dust
    radiusDUST = [0.0, 48.0, 96.0, 144.0, 192.0]
    massDUST = [1.351e+07, 7.715e+07, 1.619e+08, 2.626e+08, 3.046e+08]  # M☉



    from scipy.interpolate import interp1d




    # Interpolate cumulative stellar mass onto the dynamical mass radii
    stellar_interp = interp1d(r[valid], cumulativeStellar, bounds_error=False, fill_value='extrapolate')
    dust_interp = interp1d(radiusDUST, massDUST, bounds_error=False, fill_value='extrapolate')
    stellar_at_dyn = stellar_interp(radius_arcsec)
    dust_at_dyn = dust_interp(radius_arcsec)
    #define difference

    unseenMass = realMasses - stellar_at_dyn + dust_at_dyn
    unseenMass = np.clip(unseenMass, 0, None) #make sure dark matter doesnt fall below zero

    #to do

    #add error bars and make new plots with points (uninterpolated)
    ax_prof.plot(radius_arcsec, dust_at_dyn, color='grey', label='Dust')
    ax_prof.plot(radius_arcsec, unseenMass, color='green', label='Dark matter')
    ax_prof.plot(radius_arcsec, stellar_at_dyn, color='red', label='Stellar mass')
    ax_prof.plot(radius_arcsec, realMasses, color='blue', label='Dynamical mass')

    ax_prof.set_xlabel('Deprojected radius (arcsec)')
    ax_prof.set_ylabel('Enclosed Mass (M☉)')
    ax_prof.set_title('Cumulative Dynamical Mass')
    ax_prof.legend()
    #ax_prof.set_yscale("log")


    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Overlay plot saved to: {output_file}")
    plt.show()
    print(f"{unseenMass[-1]/realMasses[-1]} percent of this galaxy does not interact with light")








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
    keeps = [16,17]
    image = removeStars(keeps,image)
    
    
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
    print(f"{'r_mid\":':>10} {'SB (mag/pix)':>14} {'±':>4} {'N_pix':>8}")
    print("-" * 42)
    for row in results:
        print(f"{row['r_arcsec']:10.2f} {row['sb_mag']:14.4f} {row['sb_err']:6.4f} {row['n_pixels']:8d}")

    # Save to .dat
    save_profile(results, OUTPUT_FILE, sky_bg, sky_std,
                 X_CENTER, Y_CENTER, INCLINATION, POSITION_ANGLE, ZERO_POINT)

    # Plot
    plot_profile(results, output_prefix=os.path.splitext(OUTPUT_FILE)[0])
    plot_galaxy_with_profile(
    image,      
    results,
    X_CENTER, Y_CENTER,
    output_file='galaxy_profile_overlay.png'
)


