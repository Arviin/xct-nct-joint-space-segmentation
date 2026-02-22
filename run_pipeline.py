import os
import re
import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans





# =========================
# USER INPUT (placeholders)
# =========================
XCT_FOLDER = r"D:\mpc3091-Qianru\0_data\04_BdCs_Ocson\BdC02_tp0\xct_small"   # <-- put your XCT folder here
NCT_FOLDER = r"D:\mpc3091-Qianru\0_data\04_BdCs_Ocson\BdC02_tp0\nct_small"   # <-- put your NCT folder here

# If you truly don't know voxel size but they are the same, set spacing = (1,1,1)
# (Note: this is fine for registration + bivariate histogram, but not for reporting physical units.)
SPACING = (1.0, 1.0, 1.0)  # (sx, sy, sz)


# =========================
# Utilities
# =========================
import re
import os

def list_tifs(folder):
    """
    List and sort TIFF files like slice_400.tif, slice_401.tif, ...
    Ignores macOS junk files (._*).
    """
    files = []
    pattern = re.compile(r"slice_(\d+)\.(tif|tiff)$", re.IGNORECASE)

    for f in os.listdir(folder):
        if f.startswith("._"):
            continue  

        m = pattern.match(f)
        if m:
            slice_idx = int(m.group(1))
            files.append((slice_idx, f))

    if len(files) == 0:
        raise FileNotFoundError(f"No valid slice_###.tif files found in: {folder}")

    # sort by slice number
    files.sort(key=lambda x: x[0])

    # return full paths only
    return [os.path.join(folder, f) for _, f in files]


def read_stack_sitk(folder, spacing):
    """
    Reads a folder of tif slices into a 3D SimpleITK image.
    Assumes each file is a 2D slice and filename sorting defines z-order.
    """
    file_list = list_tifs(folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_list)
    img = reader.Execute()  # SITK image, size is (x,y,z)
    img = sitk.Cast(img, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    return img, file_list

def robust_normalize_sitk(img, p_lo=0.5, p_hi=99.5):
    """
    Percentile clip + scale to [0,1]. Implemented via numpy for robustness.
    """
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-12)
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.CopyInformation(img)
    return out

def make_mask_sitk(img01, thr=0.05):
    """
    Very simple mask: sample vs air in normalized space.
    For many CT datasets this is enough; you can replace with Otsu + morphology if needed.
    """
    mask = img01 > thr
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    # Optional: close small holes / remove speckles (keep minimal, robust)
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 1])
    mask = sitk.BinaryFillhole(mask)
    return mask

def center_crop_to_overlap(fixed, moving, fixed_mask=None, moving_mask=None):
    """
    If volumes differ slightly in size, crop to a common central region (safe default).
    This avoids failing when one volume is a little larger.
    """
    fx, fy, fz = fixed.GetSize()
    mx, my, mz = moving.GetSize()
    cx = min(fx, mx)
    cy = min(fy, my)
    cz = min(fz, mz)

    def crop(img, cx, cy, cz):
        sx, sy, sz = img.GetSize()
        ox = (sx - cx) // 2
        oy = (sy - cy) // 2
        oz = (sz - cz) // 2
        return sitk.RegionOfInterest(img, size=[cx, cy, cz], index=[ox, oy, oz])

    fixed_c = crop(fixed, cx, cy, cz)
    moving_c = crop(moving, cx, cy, cz)

    fixed_mask_c = crop(fixed_mask, cx, cy, cz) if fixed_mask is not None else None
    moving_mask_c = crop(moving_mask, cx, cy, cz) if moving_mask is not None else None

    return fixed_c, moving_c, fixed_mask_c, moving_mask_c

def register_rigid_mi(fixed, moving, fixed_mask=None, moving_mask=None):
    """
    Rigid (Euler3D) registration using Mattes Mutual Information (multi-modal standard).
    Returns a transform mapping moving -> fixed.
    """
    # Initial transform: align centers (geometry-based)
    initial = sitk.Euler3DTransform()
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, initial,
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    if fixed_mask is not None:
        R.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        R.SetMetricMovingMask(moving_mask)

    # Random sampling for speed/robustness
    R.SetMetricSamplingStrategy(R.NONE)
    # R.SetMetricSamplingPercentage(0.02)  # 2% (increase to 0.05 if unstable)
    # R.SetMetricSamplingSeed(123)


    R.SetInterpolator(sitk.sitkLinear)

    # Optimizer: good default for rigid MI
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    R.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.SetInitialTransform(initial, inPlace=False)

    tx = R.Execute(fixed, moving)
    return tx, R.GetMetricValue()

def resample_to_fixed(fixed, moving, transform):
    """
    Resample moving image into fixed image grid using given transform.
    """
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

def plot_qc_slices(fixed01_np, moving01_reg_np, title_prefix="QC"):
    """
    Quick QC on a central slice (z-mid): fixed, moving_reg, overlay.
    Arrays expected in (z,y,x).
    """
    z = fixed01_np.shape[0] // 2
    f = fixed01_np[z]
    m = moving01_reg_np[z]

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 3, 1); plt.imshow(f, cmap="gray"); plt.title(f"{title_prefix}: NCT (fixed)"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(m, cmap="gray"); plt.title(f"{title_prefix}: XCT resampled"); plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(f, cmap="gray")
    plt.imshow(m, cmap="magma", alpha=0.35)
    plt.title(f"{title_prefix}: overlay"); plt.axis("off")
    plt.tight_layout()
    plt.show()

def bivariate_histogram(xct01_reg_np, nct01_np, mask_np, bins=200, sample_max=2_000_000):
    """
    Compute and plot bivariate histogram: (XCT intensity, NCT intensity) on co-registered voxels.
    """
    x = xct01_reg_np[mask_np].ravel()
    y = nct01_np[mask_np].ravel()

    # Downsample for speed if huge
    if x.size > sample_max:
        idx = np.random.choice(x.size, sample_max, replace=False)
        x = x[idx]; y = y[idx]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

    plt.rcParams['image.cmap'] = 'magma'
    plt.figure(figsize=(6, 5))
    plt.imshow(
        np.log1p(H.T),
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto"
    )
    plt.xlabel("X-ray (normalized)")
    plt.ylabel("Neutron (normalized)")
    plt.title("Bivariate histogram (log count)")
    plt.colorbar(label="log(1+count)")
    plt.tight_layout()
    plt.show()


def plot_bivariate_with_marginals(
    x_xct,
    y_nct,
    *,
    bins=200,
    range_x=None,
    range_y=None,
    use_log=True,
    vmax_percentile=99.5,
    title="Bivariate histogram",
):
    """
    x_xct: 1D array of XCT intensities (registered voxels)
    y_nct: 1D array of NCT intensities (registered voxels)

    Produces: main 2D histogram + marginals on top/right.
    """

    # --- choose ranges (critical for a good plot) ---
    if range_x is None:
        range_x = np.percentile(x_xct, [0.5, 99.5])
    if range_y is None:
        range_y = np.percentile(y_nct, [0.5, 99.5])

    # --- 2D histogram: H is indexed as H[xbin, ybin] ---
    H, xedges, yedges = np.histogram2d(
        x_xct, y_nct,
        bins=bins,
        range=[range_x, range_y]
    )

    # --- build figure layout ---
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = GridSpec(
        2, 2,
        width_ratios=[4, 1.2],
        height_ratios=[1.2, 4],
        hspace=0.05, wspace=0.05
    )

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # --- marginals (1D histograms) ---
    ax_top.hist(x_xct, bins=bins, range=range_x)
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_right.hist(y_nct, bins=bins, range=range_y, orientation="horizontal")
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)

    # --- main image: MUST transpose H for correct axis meaning ---
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    if use_log:
        # log scale for counts (best for multi-phase visibility)
        # vmin=1 avoids log(0)
        im = ax_main.imshow(
            H.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(vmin=1, vmax=max(1, H.max()))
        )
        cbar_label = "Count (log scale)"
    else:
        # percentile windowing (like window/level)
        nonzero = H[H > 0]
        vmax = np.percentile(nonzero, vmax_percentile) if nonzero.size else 1
        im = ax_main.imshow(
            H.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=0, vmax=vmax
        )
        cbar_label = f"Count (vmax=p{vmax_percentile})"

    ax_main.set_xlabel("XCT intensity")
    ax_main.set_ylabel("NCT intensity")
    # ax_main.set_title(title)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)

    # nice cleanup
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()

    return H, xedges, yedges


def kmeans_one_slice(x_np, n_np, mask_np, z=None, k=4, random_state=0):
    """
    Run K-means on ONE z-slice in joint space (Ix, In), and map labels back to 2D.
    Returns:
      Z      : (N,2) feature array used for clustering
      labels : (N,) cluster labels in feature space
      lab2d  : (H,W) label image with -1 outside mask
      (x2d,n2d,m2d): the slice data
      z      : used slice index
    """
    if z is None:
        z = x_np.shape[0] // 2

    x2d = x_np[z]
    n2d = n_np[z]
    m2d = mask_np[z]

    # Feature vectors only inside the mask
    x = x2d[m2d].astype(np.float32)
    y = n2d[m2d].astype(np.float32)

    Z = np.column_stack([x, y])  # shape (N,2)

    # --- Critical: scaling ---
    # If your x and y are already in [0,1], fine.
    # If not, K-means can be dominated by the larger-range modality.
    # Here we assume your robust_normalize_sitk made them [0,1].
    # If not sure, uncomment standardization:
    # Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)

    # Map back to 2D label image
    lab2d = np.full(x2d.shape, -1, dtype=np.int16)
    lab2d[m2d] = labels.astype(np.int16)

    return Z, labels, lab2d, (x2d, n2d, m2d), z, km


def plot_kmeans_in_joint_space(Z, labels, km=None, title="K-means clusters in (XCT,NCT) space"):
    """
    Scatter plot (or density-like) of cluster labels in joint space.
    This is the 'clustered bivariate histogram' view.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=1, alpha=0.2)
    plt.xlabel("XCT intensity (registered)")
    plt.ylabel("NCT intensity (registered)")
    plt.title(title)
    if km is not None:
        c = km.cluster_centers_
        plt.scatter(c[:, 0], c[:, 1], s=120, marker="x")
    plt.tight_layout()
    plt.show()


def overlay_labels_on_slice(x2d, n2d, lab2d, title_prefix="", alpha=0.45):
    """
    Show: XCT, NCT, label map, and overlay on XCT.
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(n2d, cmap="gray")
    plt.title(f"{title_prefix} NCT slice")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(x2d, cmap="gray")
    plt.title(f"{title_prefix} XCT slice (registered)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(lab2d, cmap="tab10", vmin=-1)
    plt.title(f"{title_prefix} K-means labels")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(x2d, cmap="gray")
    plt.imshow(lab2d, cmap="tab10", alpha=alpha, vmin=-1)
    plt.title(f"{title_prefix} Labels overlay on XCT")
    plt.axis("off")

    plt.tight_layout()
    plt.show()




   




def overlay_labels_on_slice(x2d, n2d, lab2d, title_prefix="", alpha=0.45):
    """
    Show: XCT, NCT, label map, and overlay on XCT.
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(n2d, cmap="gray")
    plt.title(f"{title_prefix} NCT slice")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(x2d, cmap="gray")
    plt.title(f"{title_prefix} XCT slice (registered)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(lab2d, cmap="tab10", vmin=-1)
    plt.title(f"{title_prefix} K-means labels")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(x2d, cmap="gray")
    plt.imshow(lab2d, cmap="tab10", alpha=alpha, vmin=-1)
    plt.title(f"{title_prefix} Labels overlay on XCT")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# =========================
# Main
# =========================
def main():
    t_all = time.time()

    print("Reading stacks...")
    xct, x_files = read_stack_sitk(XCT_FOLDER, SPACING)
    nct, n_files = read_stack_sitk(NCT_FOLDER, SPACING)
    print(f"XCT size (x,y,z): {xct.GetSize()}  | files: {len(x_files)}")
    print(f"NCT size (x,y,z): {nct.GetSize()}  | files: {len(n_files)}")

    # Normalize each modality separately (good practice for MI + histogram stability)
    print("Normalizing...")
    xct01 = robust_normalize_sitk(xct)
    nct01 = robust_normalize_sitk(nct)

    # Masks (sample region only) to avoid air dominating MI
    print("Making masks...")
    x_mask = make_mask_sitk(xct01, thr=0.05)
    n_mask = make_mask_sitk(nct01, thr=0.05)

    # Crop to common overlap region if sizes differ a bit
    nct01_c, xct01_c, n_mask_c, x_mask_c = center_crop_to_overlap(nct01, xct01, n_mask, x_mask)

    # Register moving (XCT) -> fixed (NCT)
    print("Registering (rigid, MI)...")
    t0 = time.time()
    tx, metric = register_rigid_mi(nct01_c, xct01_c, fixed_mask=n_mask_c, moving_mask=x_mask_c)
    print(f"Rigid registration done in {time.time()-t0:.2f}s | MI metric: {metric:.6f}")

    # Resample XCT into NCT grid (cropped grid)
    print("Resampling XCT onto NCT grid...")
    xct01_reg = resample_to_fixed(nct01_c, xct01_c, tx)

    # QC plots
    n_np = sitk.GetArrayFromImage(nct01_c)     # (z,y,x)
    x_np = sitk.GetArrayFromImage(xct01_reg)   # (z,y,x)

    plot_qc_slices(n_np, x_np, title_prefix="Registration")

    # Combine masks in the fixed grid for bivariate histogram
    # (mask both sample regions; moving mask must be resampled too)
    x_mask_reg = sitk.Resample(x_mask_c, nct01_c, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask_np = (sitk.GetArrayFromImage(n_mask_c) > 0) & (sitk.GetArrayFromImage(x_mask_reg) > 0)

    print("Bivariate histogram...")
    bivariate_histogram(x_np, n_np, mask_np, bins=200)

    
    x = x_np[mask_np].ravel()
    y = n_np[mask_np].ravel()


    plot_bivariate_with_marginals(x, y,bins=200,use_log=False,title="XCT vs NCT bivariate histogram")




        # ---------------------------
    # K-means on ONE slice (demo)
    # ---------------------------
    z0 = x_np.shape[0] // 2   # or choose any z you want
    k = 4                     # try 4, then 6, then 8

    Z, labels, lab2d, (x2d, n2d, m2d), z_used, km = kmeans_one_slice(
        x_np, n_np, mask_np,
        z=z0, k=k, random_state=0
    )

    print(f"K-means done on slice z={z_used}, points={Z.shape[0]}, k={k}")

    # Clustered “bivariate view”
    plot_kmeans_in_joint_space(
        Z, labels, km=km,
        title=f"K-means clusters in joint space (z={z_used}, k={k})"
    )

    # Map clusters back to the slice
    overlay_labels_on_slice(
        x2d, n2d, lab2d,
        title_prefix=f"(z={z_used}, k={k})"
    )







    print(f"Total time: {time.time()-t_all:.2f}s")


if __name__ == "__main__":
    main()








