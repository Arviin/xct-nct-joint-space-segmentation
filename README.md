# XCT–NCT Joint-Space Segmentation

This repository provides a reference implementation for **unsupervised analysis of co-registered
X-ray computed tomography (XCT) and neutron computed tomography (NCT) data** using their **joint
information space**.

The code accompanies an academic manuscript currently under preparation and is intended to
support **reproducibility and methodological transparency**, rather than to serve as a general-purpose
software package.

---

## Scientific motivation

XCT and NCT provide **complementary contrast mechanisms**:

- XCT is primarily sensitive to electron density and atomic number.
- NCT is sensitive to nuclear interactions and, in many materials, light elements (e.g. hydrogen).

When XCT and NCT volumes are spatially co-registered, each voxel can be represented as a **pair of
intensity values** `(I_XCT, I_NCT)`. This defines a **joint (bivariate) intensity space**, analogous to
a mutual-information space, in which different material phases may form **distinct clusters**, even
when they are difficult to separate in either modality alone.

This repository implements a workflow that:

1. Constructs the joint intensity space from registered XCT–NCT volumes.
2. Visualizes the joint distribution using bivariate histograms.
3. Performs **unsupervised clustering (K-means)** in the joint space.
4. Maps the resulting clusters back to the spatial domain.

---

## Implemented pipeline

The current implementation performs the following steps:

1. **Input**
   - Reconstructed XCT and NCT volumes provided as stacks of 2D slices (`slice_###.tif`).
   - Volumes are assumed to correspond to approximately the same physical region.

2. **Preprocessing**
   - Robust percentile-based intensity normalization (per modality).
   - Binary masking to exclude air and background regions.

3. **Registration**
   - Rigid/Affine registration of XCT (moving) to NCT (fixed).
   - Mattes Mutual Information metric (SimpleITK).
   - Multi-resolution optimization strategy.

4. **Joint-space analysis**
   - Construction of a bivariate histogram in `(I_XCT, I_NCT)` space.
   - Optional visualization with marginal one-dimensional histograms.

5. **Unsupervised segmentation**
   - K-means clustering applied to joint-space intensity pairs.
   - Demonstration performed on a single axial slice.
   - Cluster labels are mapped back onto the registered image slice.

---

## Scope, limitations, and project status

- The current version demonstrates the method **on a single z-slice** for clarity.
- Extension to full 3D clustering or multi-slice training is conceptually straightforward but not
  included here.
- Input data are **not provided** due to size and ownership constraints, however, the data could be provided upon request.

e-mail: 
fazel.mirzaei@psi.ch ; 
seren.azad@psi.ch


In addition, this repository represents the **initial stage of a broader, ongoing research project**.
More advanced developments are currently in progress, including extensions to fully three-dimensional
analysis, improved joint-space modeling, and tighter integration with physics-informed interpretation.
These extensions will be released in future updates and documented in subsequent publications.

This code should therefore be regarded as a **demo implementation**, rather than a
fully developed end-user tool.

---

## Environment and dependencies

The computational environment is defined using **Pixi** to ensure reproducibility.

To set up the environment:

- pixi install
- pixi shell


Key dependencies include:
- Python
- NumPy
- SimpleITK
- scikit-learn
- Matplotlib

Exact package versions are recorded in `pixi.toml` and `pixi.lock`.

---

## Running the code

After activating the Pixi environment, run:

"python run_pipeline.py"


Input paths and parameters are currently specified inside the script.

---

## Author

Fazel (Arvin) Mirzaei  
Paul Scherrer Institute (PSI)  
Center for Neutron and Muon Sciences

---

## License

This code is released under the **MIT License**.  
See the `LICENSE` file for details.
