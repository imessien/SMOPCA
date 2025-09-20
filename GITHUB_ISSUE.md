# SciPy Compatibility Update for SMOPCA

## Issue Description

SMOPCA currently has compatibility issues with the latest versions of SciPy (1.13+). The main issue is with the `scipy.optimize.brentq` function which fails when the bounds don't have different signs, causing the parameter estimation to fail.

## Problem Details

1. **Root Finding Issue**: The `brentq` function in `estimateParams` method fails with "f(a) and f(b) must have different signs" error
2. **Rigid Dependencies**: Current setup.py has exact version pins that prevent compatibility with newer SciPy versions
3. **Limited Python Support**: Only supports Python 3.10, missing support for newer Python versions
4. **No CI/CD**: Missing automated testing for different SciPy versions

## Proposed Solution

### 1. Enhanced Error Handling
- Improved bounds estimation for `brentq` function
- Added fallback grid search method when `brentq` fails
- Better error handling for edge cases

### 2. Flexible Dependencies
- Updated version constraints to support SciPy 1.10.0 through 1.16+
- Made all dependencies more flexible with minimum version requirements
- Added support for Python 3.9-3.12

### 3. Comprehensive Testing
- Added test suite (`test_smopca.py`) for compatibility testing
- Tests cover basic functionality, different kernels, and edge cases
- Automated testing with GitHub Actions

### 4. CI/CD Pipeline
- Added GitHub Actions workflow for testing across Python and SciPy versions
- Automated linting and code quality checks
- Package building and validation

## Files Modified

- `src/model.py`: Enhanced error handling and bounds estimation
- `setup.py`: Updated dependencies and Python version support
- `requirements.txt`: Added for easier installation
- `test_smopca.py`: Comprehensive test suite
- `.github/workflows/ci.yml`: CI/CD pipeline
- `README.md`: Updated documentation with testing instructions

## Testing

The updated code has been tested with:
- SciPy 1.13.1 (current latest)
- Python 3.9-3.12
- All kernel types (matern, gaussian, cauchy, dummy)
- Edge cases and error conditions

## Benefits

1. **Better Compatibility**: Works with latest SciPy versions
2. **More Robust**: Handles edge cases and errors gracefully
3. **Broader Support**: Supports more Python versions
4. **Automated Testing**: CI/CD ensures compatibility across versions
5. **Better Documentation**: Clear testing and installation instructions

## Priority

**High** - This affects users trying to use SMOPCA with current SciPy installations.

## Labels

- bug
- enhancement
- compatibility
- scipy
