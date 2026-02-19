import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from parameters import name

# ── Configuration ─────────────────────────────────────────────────────────────
FITS_FILE            = name()
DETECTION_THRESHOLD  = 5.0    # sigma above background
FWHM                 = 7.0    # PSF FWHM in pixels — adjust to your image
MIN_FLUX             = 0.0    # minimum peak flux to report (set to filter faint sources)
# ─────────────────────────────────────────────────────────────────────────────

# Load image
hdul  = fits.open(FITS_FILE)
image = hdul[0].data.astype(float)
hdul.close()

print(f"Image shape: {image.shape}")

# Robust background stats
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
print(f"Background — mean: {mean:.2f}, median: {median:.2f}, std: {std:.2f}")

# Detect stars
finder  = DAOStarFinder(fwhm=FWHM, threshold=DETECTION_THRESHOLD * std)
sources = finder(image - median)

if sources is None:
    print("No sources detected. Try lowering DETECTION_THRESHOLD or FWHM.")
else:
    # Sort by peak flux, brightest first
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


