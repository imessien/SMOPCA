#!/usr/bin/env python3
"""
Test script for SMOPCA to ensure compatibility with latest SciPy versions.
"""

import numpy as np
import sys
import traceback
from src.model import SMOPCA


def test_smopca_basic():
    """Test basic SMOPCA functionality."""
    print("Testing basic SMOPCA functionality...")
    
    # Create test data
    np.random.seed(42)
    n_cells = 50
    n_genes = 100
    pos = np.random.rand(n_cells, 2)
    Y1 = np.random.rand(n_genes, n_cells)
    Y2 = np.random.rand(50, n_cells)  # protein data
    
    try:
        # Test SMOPCA initialization
        smopca = SMOPCA([Y1, Y2], pos, Z_dim=10)
        print("✓ SMOPCA initialization successful")
        
        # Test kernel building
        smopca.buildKernel(length_scale=1.0)
        print("✓ Kernel building successful")
        
        # Test parameter estimation
        smopca.estimateParams(
            iterations_gamma=1, 
            iterations_sigma_W=1, 
            sigma_init_list=[1.0, 1.0], 
            sigma_xtol_list=[1e-3, 1e-3],
            estimate_gamma=False
        )
        print("✓ Parameter estimation successful")
        
        # Test posterior calculation
        Z = smopca.calculatePosterior()
        print(f"✓ Posterior calculation successful, Z shape: {Z.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        traceback.print_exc()
        return False


def test_different_kernels():
    """Test different kernel types."""
    print("\nTesting different kernel types...")
    
    np.random.seed(42)
    n_cells = 30
    n_genes = 50
    pos = np.random.rand(n_cells, 2)
    Y1 = np.random.rand(n_genes, n_cells)
    Y2 = np.random.rand(30, n_cells)
    
    kernel_types = ['matern', 'gaussian', 'cauchy', 'dummy']
    
    for kernel_type in kernel_types:
        try:
            smopca = SMOPCA([Y1, Y2], pos, Z_dim=5, kernel_type=kernel_type)
            smopca.buildKernel(length_scale=1.0)
            print(f"✓ {kernel_type} kernel successful")
        except Exception as e:
            print(f"✗ {kernel_type} kernel failed: {e}")
            return False
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        # Test with very small data
        np.random.seed(42)
        n_cells = 10
        n_genes = 20
        pos = np.random.rand(n_cells, 2)
        Y1 = np.random.rand(n_genes, n_cells)
        Y2 = np.random.rand(10, n_cells)
        
        smopca = SMOPCA([Y1, Y2], pos, Z_dim=3)
        smopca.buildKernel(length_scale=1.0)
        smopca.estimateParams(
            iterations_gamma=1, 
            iterations_sigma_W=1, 
            sigma_init_list=[1.0, 1.0], 
            sigma_xtol_list=[1e-3, 1e-3],
            estimate_gamma=False
        )
        Z = smopca.calculatePosterior()
        print("✓ Small dataset test successful")
        
        # Test with different Z_dim
        smopca2 = SMOPCA([Y1, Y2], pos, Z_dim=1)
        smopca2.buildKernel(length_scale=1.0)
        smopca2.estimateParams(
            iterations_gamma=1, 
            iterations_sigma_W=1, 
            sigma_init_list=[1.0, 1.0], 
            sigma_xtol_list=[1e-3, 1e-3],
            estimate_gamma=False
        )
        Z2 = smopca2.calculatePosterior()
        print("✓ Z_dim=1 test successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("SMOPCA SciPy Compatibility Tests")
    print("=" * 50)
    
    # Check SciPy version
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    print()
    
    tests = [
        test_smopca_basic,
        test_different_kernels,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! SMOPCA is compatible with the current SciPy version.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
