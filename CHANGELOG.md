# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2] - 2024-12-19

### Changed
- Updated SciPy dependency from `==1.10.0` to `>=1.15.3` for compatibility with modern Python environments
- Updated all dependencies to use flexible version constraints (>=) instead of exact versions
- Added support for Python 3.11 and 3.12
- Added seaborn as an explicit dependency

### Added
- GitHub Actions workflow for automated testing across Python 3.10, 3.11, and 3.12
- Comprehensive test suite for compatibility verification
- Updated README with detailed dependency information

### Fixed
- Resolved dependency conflicts with SciPy >= 1.15.3
- Fixed brentq function call issue where bounds must have different signs
- Improved compatibility with Python 3.12 environments

## [0.1.1] - Original Release

### Added
- Initial release of SMOPCA
- Support for spatial multi-omics dimension reduction
- Integration with scanpy and anndata
