# XCT–NCT Joint-Space Segmentation

This repository provides a demonstration workflow for the **unsupervised analysis of co-registered
X-ray computed tomography (XCT) and neutron computed tomography (NCT) data** in their
**joint attenuation space**.

The workflow accompanies the peer-reviewed publication:

> **F. Mirzaei, L. El Arab, E. Granget, L. Cristina, N. Diomidis,
> L. Brambilla, D. C. Mannes, M. Strobl, A. Kaestner, and S. Azad**  
> *Unsupervised 3D segmentation and statistical corrosion-layer stratigraphy of
> long-term corrosion in carbon steel archaeological analogs using co-registered
> bimodal tomograms*  
> **npj Materials Degradation** (2026)  
> **DOI:** [10.1038/s41529-026-00882-w](https://doi.org/10.1038/s41529-026-00882-w)  
> **Article:** [Nature / npj Materials Degradation](https://www.nature.com/articles/s41529-026-00882-w)

The repository is intended to support **reproducibility, methodological transparency,
and reuse of the analysis workflow**, rather than to serve as a general-purpose
segmentation software package.

---

## Scientific motivation

X-ray and neutron computed tomography provide complementary interaction mechanisms
and therefore complementary material contrast.

### X-ray computed tomography

In XCT, typically using photon energies from tens to a few hundred keV, image
contrast is governed by the energy-dependent linear attenuation coefficient
$\mu_X(E)$.

Within this energy range, attenuation is dominated primarily by photoelectric
absorption and Compton scattering. XCT therefore generally provides strong contrast
for dense and high-effective-atomic-number materials and is particularly effective
for resolving the geometry of metallic components and dense corrosion products.

### Neutron computed tomography

For cold and thermal neutrons, transmission contrast is governed by the macroscopic
total cross-section

$$
\Sigma_t(E)=N\sigma_t(E),
$$

where $N$ is the atomic number density and $\sigma_t(E)$ includes neutron absorption
and scattering contributions that remove neutrons from the transmitted beam.

Unlike X-ray attenuation, neutron interaction cross-sections are isotope-specific
and do not vary monotonically with atomic number. Neutron imaging can therefore
provide strong contrast for materials or elements that are comparatively difficult
to distinguish using XCT alone, particularly hydrogen-containing and hydrated
materials.

---

## Joint neutron–X-ray attenuation space

After geometric registration, the NCT and XCT volumes are defined on a common voxel
grid. Each spatial coordinate $\mathbf{x}$ can therefore be represented by a paired
feature vector

$$
\mathbf{f}(\mathbf{x}) =
\left(
I_{\mathrm{NCT}}(\mathbf{x}),
I_{\mathrm{XCT}}(\mathbf{x})
\right).
$$

The collection of these voxel-wise intensity pairs defines a two-dimensional
**joint neutron–X-ray attenuation space**.

Materials or structural domains that overlap substantially in one imaging modality
may occupy more distinct regions when both attenuation measurements are considered
simultaneously. The resulting joint distribution can therefore provide improved
discrimination of attenuation-defined domains compared with either modality alone.

Importantly, clusters identified in this feature space should not automatically be
interpreted as individual material or mineral phases. Their physical interpretation
requires consideration of their spatial distribution in the reconstructed volume and,
where appropriate, complementary chemical or structural measurements.

---

## Workflow

The general analysis workflow consists of:

1. **Pre-processing and normalization** of reconstructed XCT and NCT volumes.
2. **Multimodal image registration** to establish voxel-wise spatial correspondence.
3. Construction of the **joint neutron–X-ray attenuation space**.
4. Visualization using **bivariate intensity histograms**.
5. **Unsupervised K-means clustering** of voxel-wise XCT–NCT feature vectors.
6. Mapping of cluster assignments back into the tomographic spatial domain.
7. Spatial interpretation and comparison of the resulting attenuation-defined regions.

In the associated publication, the methodology was applied to co-registered neutron
and X-ray tomograms of archaeological iron specimens. K-means model orders
$k=4$, $k=6$, and $k=8$ were used as controlled levels of segmentation granularity
to investigate the reproducibility of corrosion-related attenuation domains across
multiple specimens.

The cluster identities were interpreted only after mapping the attenuation-space
partition back into tomogram space. Accordingly, the method distinguishes between
the **unsupervised statistical partitioning of the joint feature space** and the
subsequent **spatially informed interpretation of the resulting domains**.

---

## Implemented pipeline

The current demonstration implementation performs the following steps:

### 1. Input

- Reconstructed XCT and NCT volumes provided as stacks of 2D image slices.
- The two datasets should represent approximately the same physical specimen region.

### 2. Pre-processing

- Robust percentile-based intensity normalization.
- Binary masking to exclude air and background regions.
- Preparation of the reconstructed volumes for multimodal registration and
  joint-space analysis.

### 3. Registration

- Rigid and affine multimodal registration.
- Mattes Mutual Information similarity metric implemented using SimpleITK.
- Multi-resolution optimization strategy.
- Resampling onto a common spatial grid to establish voxel-wise correspondence.

### 4. Joint-space analysis

- Construction of a bivariate histogram in `(I_XCT, I_NCT)` space.
- Visualization of the joint intensity distribution.
- Optional visualization of the corresponding one-dimensional marginal
  distributions.

### 5. Unsupervised segmentation

- K-means clustering applied to paired XCT–NCT intensity features.
- Cluster assignments mapped back to the corresponding spatial coordinates.
- Visualization of attenuation-defined domains in image space.

The demonstration code is intentionally compact and is designed primarily to expose
the methodological steps clearly.

---

## Scope, limitations, and project status

This repository is a **demonstration implementation** of the analysis concepts used
in the associated publication rather than a general-purpose segmentation package.

The current demonstration emphasizes a simplified workflow for clarity. In
particular:

- The supplied example demonstrates clustering and visualization on selected image
  data rather than reproducing every analysis performed for the full publication.
- Image contrast, registration accuracy, reconstruction artifacts, noise,
  spatial resolution, and partial-volume effects can influence the geometry of the
  joint feature space and the resulting clusters.
- K-means partitions attenuation space; the resulting clusters should not
  automatically be interpreted as distinct chemical or mineralogical phases.
- Physical interpretation should be supported by spatial information and, where
  available, independent material-characterization measurements.
- Input tomography datasets are not distributed through this repository because of
  their size and institutional data-sharing constraints. Data associated with the
  publication are available upon reasonable request.

This repository also represents part of a broader, ongoing research effort.
Further developments include extension of the methodology to fully three-dimensional
analysis, alternative joint-space models, quantitative cross-specimen comparison,
and stronger integration of spatial and physics-informed information.

---

## Citation

If you use this workflow, code, or methodology in academic work, please cite the
associated publication:

**Mirzaei, F., El Arab, L., Granget, E., Cristina, L., Diomidis, N.,
Brambilla, L., Mannes, D. C., Strobl, M., Kaestner, A., & Azad, S. (2026).**  
*Unsupervised 3D segmentation and statistical corrosion-layer stratigraphy of
long-term corrosion in carbon steel archaeological analogs using co-registered
bimodal tomograms.*  
**npj Materials Degradation.**  
[https://doi.org/10.1038/s41529-026-00882-w](https://doi.org/10.1038/s41529-026-00882-w)

Publisher page:  
[https://www.nature.com/articles/s41529-026-00882-w](https://www.nature.com/articles/s41529-026-00882-w)

### BibTeX

```bibtex
@article{Mirzaei2026NXCT,
  author = {Mirzaei, Fazel and
            El Arab, Lara and
            Granget, Elodie and
            Cristina, Laura and
            Diomidis, Nikitas and
            Brambilla, Laura and
            Mannes, David Christian and
            Strobl, Markus and
            Kaestner, Anders and
            Azad, Seren},
  title = {Unsupervised 3D segmentation and statistical corrosion-layer
           stratigraphy of long-term corrosion in carbon steel archaeological
           analogs using co-registered bimodal tomograms},
  journal = {npj Materials Degradation},
  year = {2026},
  doi = {10.1038/s41529-026-00882-w},
  url = {https://doi.org/10.1038/s41529-026-00882-w}
}
```

---

## Environment and dependencies

The computational environment is defined using **Pixi** to improve reproducibility.

Install the environment with:

```bash
pixi install
```

Activate it with:

```bash
pixi shell
```

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

```bash
python run_pipeline.py
```

Input paths and analysis parameters are currently specified inside the script.

---

## Data availability

The tomography datasets are not included in this repository because of their size
and institutional data-sharing restrictions.

Data associated with the published study, including raw and processed tomography
data, are available upon reasonable request.

For questions regarding the publication, data, or analysis workflow, contact:

- **Fazel Mirzaei:** fazel.mirzaei@psi.ch
- **Seren Azad:** seren.azad@psi.ch

---

## Author

**Fazel (Arvin) Mirzaei**  
Center for Neutron and Muon Sciences  
Paul Scherrer Institute (PSI)  
Villigen, Switzerland

---

## License

This code is released under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.
