import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import IRAFStarFinder
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from parameters import names
import fitsHeaderGetter
import astroSearch

from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

def measure_fwhm(image, n_stars=20):
    """
    Fit 2D Gaussians to the brightest isolated stars and return median FWHM.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    # Use a rough initial finder with a guess FWHM
    finder  = DAOStarFinder(fwhm=5.0, threshold=10.*std)
    sources = finder(image - median)
    sources.sort('peak')
    sources.reverse()

    fwhms = []
    fit_g = fitting.LevMarLSQFitter()

    for s in sources[:n_stars]:  # only use brightest n_stars
        x0, y0 = int(s['xcentroid']), int(s['ycentroid'])
        box = 20  # cutout half-size in pixels

        # Extract small cutout around star
        x_lo, x_hi = max(0, x0-box), min(image.shape[1], x0+box)
        y_lo, y_hi = max(0, y0-box), min(image.shape[0], y0+box)
        cutout = image[y_lo:y_hi, x_lo:x_hi] - median

        yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]

        # Fit a 2D Gaussian
        g_init = models.Gaussian2D(amplitude=cutout.max(),
                                   x_mean=box, y_mean=box,
                                   x_stddev=3, y_stddev=3)
        try:
            g_fit = fit_g(g_init, xx, yy, cutout)
            # FWHM = 2.355 * sigma for a Gaussian
            fwhm_x = 2.355 * g_fit.x_stddev.value
            fwhm_y = 2.355 * g_fit.y_stddev.value
            fwhm   = (fwhm_x + fwhm_y) / 2.0

            if 1.0 < fwhm < 30.0:  # sanity check — reject bad fits
                fwhms.append(fwhm)
                print(f"  Star at ({x0},{y0}): FWHM = {fwhm:.2f} px")
        except Exception:
            continue

    if not fwhms:
        print("No good fits found, defaulting to 5.0 px")
        return 5.0

    measured = float(np.median(fwhms))
    print(f"\nMedian FWHM across {len(fwhms)} stars: {measured:.2f} px")
    return measured
# ── Configuration ─────────────────────────────────────────────────────────────
galaxyName, FITS_FILE,notImportant            = names()
DETECTION_THRESHOLD  = 4.0    # sigma above background
#FWHM = measure_fwhm(FITS_FILE)    # PSF FWHM in pixels — adjust to your image
MIN_FLUX             = 0.0    # minimum peak flux to report (set to filter faint sources)
# ─────────────────────────────────────────────────────────────────────────────

# Load image
hdul  = fits.open(FITS_FILE)
image = hdul[0].data.astype(float)
hdul.close()

FWHM = measure_fwhm(image)

print(f"Image shape: {image.shape}")

#  background stats
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
print(f"Background — mean: {mean:.2f}, median: {median:.2f}, std: {std:.2f}")

# Detect stars
finder  = IRAFStarFinder(fwhm=FWHM, threshold=DETECTION_THRESHOLD * std)
sources = finder(image - median)

if sources is None:
    print("No sources detected. Try lowering DETECTION_THRESHOLD or FWHM.")
else:
    # Sort by peak flux, brightest first
    
    mask = np.ones(len(sources), dtype=bool)

    for i, s in enumerate(sources):
        x = s['xcentroid']
        y = s['ycentroid']
        RA, dec = fitsHeaderGetter.pixelCoordinatesToRADEC(x, y)
        if not astroSearch.MWchecker(RA, dec): # returns true if star is in mw
            mask[i] = False
            print()

    sources = sources[mask]
            



    sources.sort('peak')
    sources.reverse()
    
    print(f"\nDetected {len(sources)} sources:\n")
    print(f"{'ID':>4}  {'x (px)':>8}  {'y (px)':>8}  {'peak flux':>12}  {'sharpness':>10}")
    print("-" * 50)
    for s in sources:
        print(f"{s['id']:>4}  {s['xcentroid']:8.2f}  {s['ycentroid']:8.2f}  "
              f"{s['peak']:12.2f}  {s['sharpness']:10.4f}")
        
    

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.nanpercentile(image, [1, 99])
    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

    for s in sources:
        # Circle size scales with peak flux
        radius = 5 + 10 * (s['peak'] - sources['peak'].min()) / \
                 (sources['peak'].max() - sources['peak'].min() + 1e-9)
        circle = plt.Circle((s['xcentroid'], s['ycentroid']), radius,
                             color='red', fill=False, linewidth=1.0)
        ax.add_patch(circle)
        ax.text(s['xcentroid'] + radius + 2, s['ycentroid'],
                f"#{s['id']}  ({s['xcentroid']:.0f}, {s['ycentroid']:.0f})\n"
                f"peak: {s['peak']:.0f}",
                color='red', fontsize=6, va='center')

    ax.set_title(f"Detected sources (threshold={DETECTION_THRESHOLD}σ, FWHM={FWHM}px)")
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    plt.tight_layout()
    plt.savefig('detected_stars.png', dpi=150)
    print("\nPlot saved to detected_stars.png")
    plt.show()

    # Save source list to file
    sources.write('detected_stars.dat', format='ascii.commented_header', overwrite=True)
    print("Source list saved to detected_stars.dat")


