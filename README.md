# Spatial Multi-Omics PCA

Spatial Multi-Omics PCA (SMOPCA) is a novel dimension reduction method to integrate multi-modal data and extract low-dimensional representations with preserved spatial dependencies among spots.

![fig1](./img/fig1.png)

## Installation

SMOPCA can be installed directly from PyPI using the following command:
```bash
pip install SMOPCA
```

Or install from source:
```bash
git clone https://github.com/cmhimself/SMOPCA.git
cd SMOPCA
pip install -e .
```

### Requirements

- Python >= 3.9
- SciPy >= 1.10.0 (compatible with latest versions up to 1.16+)
- NumPy >= 1.23.0
- scikit-learn >= 1.2.0
- scanpy >= 1.9.0
- pandas >= 1.5.0
- matplotlib >= 3.6.0
- anndata >= 0.8.0

## Run SMOPCA

1. Prepare input data
   - SMOPCA accepts gene expression and protein/atac data matrices. Each modality of the data is preprocessed and normalized  separately. Initially, genes and proteins with zero counts were filtered out. Subsequently, the count matrix was normalized based on library size, followed by log-transformation and scaling to achieve unit variance and zero mean. ATAC reads are mapped to gene regions and the peak matrix is collapsed into a gene activity matrix, adhering to the established protocol from the Satija lab. The gene activity matrix was preprocessed and normalized using the same method as applied to mRNA data. Finally, we recommend to save the data into a hdf5 file.
   - Note that SMOPCA takes input matrices with columns corresponding to cells or spots.
2. Specify model hyperparameters and Model training
   - the dimensionality of the latent factors (default 20)
   - the kernel type (default matern kernel)
   - For the rest of the parameters, see more in tutorials.
3. Downstream analysis
   - Visualization
   - Clustering analysis
   - Differential expression analysis
   - GSEA
   - Other tasks

## Testing

To test SMOPCA installation and compatibility with your SciPy version:

```bash
python test_smopca.py
```

This will run comprehensive tests including:
- Basic functionality tests
- Different kernel type tests
- Edge case handling
- SciPy version compatibility

## Datasets

Sample datasets are provided in ./data folder. The rest of the datasets used in this study are available at https://doi.org/10.5281/zenodo.15187362

## Changelog

### v0.1.2 (Latest)
- **SciPy Compatibility**: Updated to work with SciPy versions 1.10.0 through 1.16+
- **Improved Error Handling**: Enhanced robustness for root finding algorithms
- **Flexible Dependencies**: Updated version constraints for better compatibility
- **CI/CD Pipeline**: Added GitHub Actions for automated testing
- **Python Support**: Extended support to Python 3.9-3.12
