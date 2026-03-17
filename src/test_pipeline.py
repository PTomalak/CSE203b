import unittest
import numpy as np
import sys
import os
import io
from contextlib import redirect_stdout

from loader import load_ply, normalize_theta_params
from fast_solver import generate_cameras_numpy, compute_fisher_information_numpy, solve_d_optimal_frank_wolfe_numpy

class TestExperimentPipeline(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        np.random.seed(42)
        
    def test_camera_generation(self):
        """Verifies candidate cameras are generated correctly on a hemisphere (Testing pure Numpy logic)."""
        M = 50
        radius = 0.3
        
        with redirect_stdout(io.StringIO()):
            cameras = generate_cameras_numpy(num_cameras=M, radius=radius)
            
        self.assertEqual(len(cameras), M)
        for cam in cameras:
            # Check position radius
            pos = cam['pos']
            self.assertAlmostEqual(np.linalg.norm(pos), radius, places=4)
            # Check cameras are strictly on the hemisphere spanning Z > 0
            self.assertTrue(pos[2] >= 0)
            # Check orthogonality of R: R^T R = I
            R = cam['R']
            I = np.eye(3)
            self.assertTrue(np.allclose(R.T @ R, I, atol=1e-4))
            
    def test_fisher_info_shapes_and_psd(self):
        """Tests that the extracted geometric Jacobians form PSD Fisher matrices via Numpy engine."""
        N = 20
        M = 10
        # mu: (N, 3), scale: (N, 3), quat: (N, 4)
        mu = np.random.randn(N, 3)
        scale = np.log(np.random.rand(N, 3) * 0.1 + 0.01)
        quat = np.random.randn(N, 4)
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        theta_params = np.concatenate([mu, scale, quat], axis=1)
        
        with redirect_stdout(io.StringIO()):
            cameras = generate_cameras_numpy(num_cameras=M, radius=0.3)
            F_blocks = compute_fisher_information_numpy(theta_params, cameras)
            
        # Check shape is (M, N, 10, 10)
        self.assertEqual(F_blocks.shape, (M, N, 10, 10))
        
        # Check that each block is positive semi-definite (eigenvalues >= -1e-5)
        for j in range(M):
            for i in range(N):
                F = F_blocks[j, i]
                # F = J^T J, so it should be symmetric PSD
                self.assertTrue(np.allclose(F, F.T, atol=1e-5))
                eigvals = np.linalg.eigvalsh(F)
                self.assertTrue(np.all(eigvals >= -1e-4), msg=f"Found negative eigenvalue {eigvals.min()} in F_blocks")
                
    def test_solve_frank_wolfe(self):
        """Verifies D-optimal solver convergence, structure, and constraints."""
        N = 30
        M = 20
        K = 5
        
        # Create random PSD blocks for F_blocks
        F_blocks = np.random.randn(M, N, 10, 10)
        F_blocks = np.einsum('mnab, mncb -> mnac', F_blocks, F_blocks) # J^T J
        
        with redirect_stdout(io.StringIO()):
            w_star, history = solve_d_optimal_frank_wolfe_numpy(F_blocks, K=K, max_iter=50, lambda_reg=1.0)
            
        # Check weights sum to exactly K
        self.assertAlmostEqual(np.sum(w_star), K, places=4)
        
        # Frank-Wolfe domain check: w_i >= 0 and w_i <= 1 (approx)
        self.assertTrue(np.all(w_star >= -1e-5))
        self.assertTrue(np.all(w_star <= 1.0 + 1e-5))
        
        # Verify Block Diagonal Trace matches Dense Trace exactly
        M_w = np.einsum('m, mnab -> nab', w_star, F_blocks) + 1.0 * np.expand_dims(np.eye(10), 0).repeat(N, axis=0)
        block_inv = np.linalg.inv(M_w)
        trace_block = np.einsum('nii->', block_inv)
        
        dense_M = np.zeros((10*N, 10*N))
        for i in range(N):
            dense_M[10*i:10*(i+1), 10*i:10*(i+1)] = M_w[i]
            
        dense_inv = np.linalg.inv(dense_M)
        trace_dense = np.trace(dense_inv)
        
        self.assertAlmostEqual(trace_block, trace_dense, places=2)
        
    def test_real_models_robustness(self):
        """Tests the full pipeline on a subset of downloaded .ply models across multiple varied configurations."""
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
        if not os.path.exists(models_dir):
            self.skipTest("Models directory not found. Skipping real data test.")
            
        ply_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith("point_cloud.ply"):
                    ply_files.append(os.path.join(root, file))
        
        # Stop early if no models
        if not ply_files:
            self.skipTest("No point_cloud.ply files found in models directory.")
            
        # Test all models, but use just their iteration_30000 final point cloud to save time
        test_files = [f for f in ply_files if 'iteration_30000' in f]
        
        configs = [
            (50, 20, 5, 1e-2, 2),
            (100, 50, 10, 1e-3, 2),
            (20, 10, 2, 1e-1, 2),
            (200, 100, 20, 1e-2, 2),
        ]
        
        print("\n\n" + "="*80)
        print(f"REAL MODEL ROBUSTNESS SWEEP SUMMARY")
        print("="*105)
        print(f"{'Model Name':<15} | {'Splats':<8} | {'Cameras':<8} | {'Budget K':<8} | {'λ_frac':<8} | {'Result':<8} | {'Validation Details'}")
        print("-" * 105)
        
        for ply_file in test_files:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ply_file))))
            
            for max_splats, M, K, lam_frac, radius in configs:
                with self.subTest(model=model_name, splats=max_splats, cameras=M, budget=K, lam_frac=lam_frac, radius=radius):
                    with redirect_stdout(io.StringIO()):
                        try:
                            theta_params, _ = load_ply(ply_file, max_splats=max_splats, align=True, flip=True)
                            theta_params = normalize_theta_params(theta_params)
                        except Exception as e:
                            self.fail(f"Failed to load {ply_file}: {e}")
                            
                        cameras = generate_cameras_numpy(num_cameras=M, radius=radius)
                        F_blocks = compute_fisher_information_numpy(theta_params, cameras)
                        w_star, history = solve_d_optimal_frank_wolfe_numpy(F_blocks, K=K, max_iter=20, lambda_frac=lam_frac)
                        
                        out_dir = os.path.join(os.path.dirname(__file__), 'test_out')
                        os.makedirs(out_dir, exist_ok=True)
                        plot_filename = f"{model_name}_splats{max_splats}_cams{M}_budget{K}_lfrac{lam_frac}_rad{radius}.png"
                        plot_path = os.path.join(out_dir, plot_filename)
                        # Import visualize_results from local for plotting
                        from local import visualize_results
                        visualize_results(theta_params, cameras, w_star, K, plot_path)
                        
                    self.assertAlmostEqual(np.sum(w_star), K, places=3)
                    self.assertTrue(np.all(np.isfinite(w_star)))
                    self.assertTrue(np.all(w_star >= -1e-4), "Found negative weights")
                    self.assertTrue(np.all(w_star <= K + 1e-4), "Weights exceeded budget bound")
                    
                    reason = f"Sum(w)={np.sum(w_star):.2f}=={K}, w∈[0,K]"
                    print(f"{model_name:<15} | {max_splats:<8} | {M:<8} | {K:<8} | {lam_frac:<8} | {'PASSED':<8} | {reason}")
                    
        print("="*105 + "\n")

if __name__ == '__main__':
    unittest.main(verbosity=1)
