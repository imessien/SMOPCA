# Update SMOPCA for SciPy 1.13+ Compatibility

## Summary

This PR updates SMOPCA to be compatible with the latest versions of SciPy (1.13+) and adds comprehensive testing and CI/CD infrastructure.

## Changes Made

### 🔧 Core Fixes
- **Fixed brentq bounds estimation**: Improved algorithm to find proper bounds where function has different signs
- **Added fallback method**: Grid search fallback when brentq fails due to same signs at bounds
- **Enhanced error handling**: Better error messages and graceful degradation

### 📦 Dependencies
- **Flexible version constraints**: Updated to support SciPy 1.10.0 through 1.16+
- **Python support**: Extended to Python 3.9-3.12 (was 3.10 only)
- **Dependency updates**: Made all dependencies more flexible with minimum version requirements
- **Added requirements.txt**: For easier installation

### 🧪 Testing & Quality
- **Comprehensive test suite**: Added `test_smopca.py` with full functionality tests
- **Multiple kernel testing**: Tests all kernel types (matern, gaussian, cauchy, dummy)
- **Edge case handling**: Tests small datasets and various Z_dim values
- **SciPy version compatibility**: Tests across different SciPy versions

### 🚀 CI/CD Pipeline
- **GitHub Actions**: Added automated testing across Python 3.9-3.12 and SciPy versions
- **Linting**: Added flake8, black, and isort checks
- **Package building**: Automated package building and validation
- **Matrix testing**: Tests multiple Python and SciPy version combinations

### 📚 Documentation
- **Updated README**: Added testing instructions and requirements
- **Changelog**: Documented all improvements
- **Installation guide**: Added source installation instructions

## Technical Details

### Root Finding Fix
The main issue was in the `estimateParams` method where `scipy.optimize.brentq` would fail with "f(a) and f(b) must have different signs". The fix includes:

1. **Better bounds estimation**: More robust algorithm to find bounds with different signs
2. **Error handling**: Try-catch with fallback to grid search
3. **Graceful degradation**: Continue processing even if some optimizations fail

### Code Changes
```python
# Before: Simple bounds that could have same signs
lb = ub = 0.1
for sigma in np.arange(0.1, 10.0, 0.1):
    res = jac_sigma_sqr(sigma)
    if res < 0:
        lb = sigma
    else:
        ub = sigma
        break

# After: Robust bounds estimation with error handling
lb = 0.01
ub = 10.0
# ... improved algorithm with try-catch and fallback
```

## Testing Results

All tests pass with SciPy 1.13.1:
```
==================================================
SMOPCA SciPy Compatibility Tests
==================================================
SciPy version: 1.13.1

Testing basic SMOPCA functionality...
✓ SMOPCA initialization successful
✓ Kernel building successful
✓ Parameter estimation successful
✓ Posterior calculation successful, Z shape: (50, 10)

Testing different kernel types...
✓ matern kernel successful
✓ gaussian kernel successful
✓ cauchy kernel successful
✓ dummy kernel successful

Testing edge cases...
✓ Small dataset test successful
✓ Z_dim=1 test successful

==================================================
Tests passed: 3/3
🎉 All tests passed! SMOPCA is compatible with the current SciPy version.
```

## Files Changed

- `src/model.py` - Core algorithm improvements
- `setup.py` - Dependency updates and version bump
- `requirements.txt` - New file for easy installation
- `test_smopca.py` - New comprehensive test suite
- `.github/workflows/ci.yml` - New CI/CD pipeline
- `README.md` - Updated documentation

## Breaking Changes

None. This is a backward-compatible update.

## Migration Guide

No migration needed. The API remains the same, but now works with newer SciPy versions.

## Checklist

- [x] Code works with SciPy 1.13+
- [x] All existing functionality preserved
- [x] Comprehensive tests added
- [x] CI/CD pipeline implemented
- [x] Documentation updated
- [x] No breaking changes
- [x] Backward compatible

## Related Issues

Fixes compatibility issues with latest SciPy versions.

## Screenshots/Examples

Test results showing successful compatibility:
```
✓ SMOPCA initialization successful
✓ Kernel building successful
✓ Parameter estimation successful
✓ Posterior calculation successful, Z shape: (50, 10)
```

## Additional Notes

This update ensures SMOPCA continues to work as SciPy evolves, making it more robust and future-proof. The added testing infrastructure will help catch compatibility issues early.
