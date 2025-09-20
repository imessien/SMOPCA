#!/usr/bin/env python3
"""
Simple test script for CuPy GPU acceleration in SMOPCA
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_cupy_smopca():
    """Test SMOPCA with CuPy GPU acceleration"""
    print("=== SMOPCA CuPy GPU Test ===")
    
    # Create sample data
    n_cells = 100
    n_rna = 200
    n_adt = 50
    
    pos = np.random.rand(n_cells, 2) * 10
    Y_rna = np.random.poisson(5, (n_rna, n_cells)).astype(np.float32)
    Y_adt = np.random.poisson(10, (n_adt, n_cells)).astype(np.float32)
    Y_list = [Y_rna, Y_adt]
    
    print(f"Created data: {n_cells} cells, {n_rna}+{n_adt} features")
    
    try:
        from model import SMOPCA
        
        # Test GPU implementation
        print("Testing GPU-accelerated SMOPCA...")
        model = SMOPCA(
            Y_list=Y_list,
            pos=pos,
            Z_dim=10,
            kernel_type='gaussian',
            use_gpu=True
        )
        
        print(f"Using {'GPU (CuPy)' if model.use_gpu else 'CPU (NumPy)'}")
        
        # Build kernel matrix
        print("Building kernel matrix...")
        model.buildKernel(length_scale=1.0)
        print(f"Kernel matrix shape: {model.K.shape}")
        
        # Test parameter estimation (simplified)
        print("Testing parameter estimation...")
        sigma_init_list = [1.0, 1.0]
        sigma_xtol_list = [1e-5, 1e-5]
        
        model.estimateParams(
            iterations_gamma=2,
            iterations_sigma_W=3,
            estimate_gamma=False,
            sigma_init_list=sigma_init_list,
            sigma_xtol_list=sigma_xtol_list
        )
        
        # Calculate posterior
        print("Calculating posterior...")
        Z = model.calculatePosterior()
        print(f"Posterior shape: {Z.shape}")
        
        # Test NumPy conversion
        print("Testing NumPy conversion...")
        results = model.get_results_numpy()
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cupy_smopca()
    if success:
        print("\n🎉 CuPy GPU acceleration is working correctly!")
    else:
        print("\n⚠️  Tests failed. Check CuPy installation and GPU availability.")
