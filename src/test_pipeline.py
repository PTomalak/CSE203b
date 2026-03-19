import io
import json
import os
import unittest
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np

from loader import load_ply, normalize_theta_params
from fast_solver import (
    compute_fisher_information_numpy,
    continuous_relaxed_objective,
    discrete_subset_objective,
    generate_cameras_numpy,
    integrality_gap_percent,
    round_pipage,
    round_randomized_best_of_N,
    round_swap_local_search,
    round_topK,
    solve_d_optimal_frank_wolfe_numpy,
)


class TestExperimentPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_camera_generation(self):
        M = 50
        radius = 0.3
        with redirect_stdout(io.StringIO()):
            cameras = generate_cameras_numpy(num_cameras=M, radius=radius)
        self.assertEqual(len(cameras), M)
        for cam in cameras:
            pos = cam['pos']
            self.assertAlmostEqual(np.linalg.norm(pos), radius, places=4)
            self.assertTrue(pos[2] >= 0)
            self.assertTrue(np.allclose(cam['R'].T @ cam['R'], np.eye(3), atol=1e-4))

    def test_fisher_info_shapes_and_psd(self):
        N = 20
        M = 10
        mu = np.random.randn(N, 3)
        scale = np.log(np.random.rand(N, 3) * 0.1 + 0.01)
        quat = np.random.randn(N, 4)
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        theta_params = np.concatenate([mu, scale, quat], axis=1)

        with redirect_stdout(io.StringIO()):
            cameras = generate_cameras_numpy(num_cameras=M, radius=0.3)
            F_blocks = compute_fisher_information_numpy(theta_params, cameras)

        self.assertEqual(F_blocks.shape, (M, N, 10, 10))
        for j in range(M):
            for i in range(N):
                F = F_blocks[j, i]
                self.assertTrue(np.allclose(F, F.T, atol=1e-5))
                eigvals = np.linalg.eigvalsh(F)
                self.assertTrue(np.all(eigvals >= -1e-4), msg=f"Found negative eigenvalue {eigvals.min()} in F_blocks")

    def test_solve_frank_wolfe(self):
        N = 30
        M = 20
        K = 5
        F_blocks = np.random.randn(M, N, 10, 10)
        F_blocks = np.einsum('mnab, mncb -> mnac', F_blocks, F_blocks)

        with redirect_stdout(io.StringIO()):
            w_star, history = solve_d_optimal_frank_wolfe_numpy(F_blocks, K=K, max_iter=50, lambda_reg=1.0)

        self.assertAlmostEqual(np.sum(w_star), K, places=4)
        self.assertTrue(np.all(w_star >= -1e-5))
        self.assertTrue(np.all(w_star <= 1.0 + 1e-5))

        M_w = np.einsum('m, mnab -> nab', w_star, F_blocks) + 1.0 * np.expand_dims(np.eye(10), 0).repeat(N, axis=0)
        block_inv = np.linalg.inv(M_w)
        trace_block = np.einsum('nii->', block_inv)

        dense_M = np.zeros((10 * N, 10 * N))
        for i in range(N):
            dense_M[10 * i:10 * (i + 1), 10 * i:10 * (i + 1)] = M_w[i]

        dense_inv = np.linalg.inv(dense_M)
        trace_dense = np.trace(dense_inv)
        self.assertAlmostEqual(trace_block, trace_dense, places=2)

    def test_real_models_robustness(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
        if not os.path.exists(models_dir):
            self.skipTest("Models directory not found. Skipping real data test.")

        ply_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith("point_cloud.ply"):
                    ply_files.append(os.path.join(root, file))

        if not ply_files:
            self.skipTest("No point_cloud.ply files found in models directory.")

        test_files = [f for f in ply_files if 'iteration_30000' in f]
        configs = [
            (50, 20, 5, 1e-2, 2),
            (100, 50, 10, 1e-3, 2),
            (20, 10, 2, 1e-1, 2),
            (200, 100, 20, 1e-2, 2),
        ]

        print("\n\n" + "=" * 80)
        print("REAL MODEL ROBUSTNESS SWEEP SUMMARY")
        print("=" * 105)
        print(f"{'Model Name':<15} | {'Splats':<8} | {'Cameras':<8} | {'Budget K':<8} | {'λ_frac':<8} | {'Result':<8} | {'Validation Details'}")
        print("-" * 105)

        for ply_file in test_files:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ply_file))))
            for max_splats, M, K, lam_frac, radius in configs:
                with self.subTest(model=model_name, splats=max_splats, cameras=M, budget=K, lam_frac=lam_frac, radius=radius):
                    with redirect_stdout(io.StringIO()):
                        theta_params, _ = load_ply(ply_file, max_splats=max_splats, align=True, flip=True)
                        theta_params = normalize_theta_params(theta_params)
                        cameras = generate_cameras_numpy(num_cameras=M, radius=radius)
                        F_blocks = compute_fisher_information_numpy(theta_params, cameras)
                        w_star, history = solve_d_optimal_frank_wolfe_numpy(F_blocks, K=K, max_iter=20, lambda_frac=lam_frac)
                        out_dir = os.path.join(os.path.dirname(__file__), 'test_out')
                        os.makedirs(out_dir, exist_ok=True)
                        plot_filename = f"{model_name}_splats{max_splats}_cams{M}_budget{K}_lfrac{lam_frac}_rad{radius}.png"
                        plot_path = os.path.join(out_dir, plot_filename)
                        from local import visualize_results
                        visualize_results(theta_params, cameras, w_star, K, plot_path)

                    self.assertAlmostEqual(np.sum(w_star), K, places=3)
                    self.assertTrue(np.all(np.isfinite(w_star)))
                    self.assertTrue(np.all(w_star >= -1e-4), "Found negative weights")
                    self.assertTrue(np.all(w_star <= K + 1e-4), "Weights exceeded budget bound")
                    reason = f"Sum(w)={np.sum(w_star):.2f}=={K}, w∈[0,K]"
                    print(f"{model_name:<15} | {max_splats:<8} | {M:<8} | {K:<8} | {lam_frac:<8} | {'PASSED':<8} | {reason}")

        print("=" * 105 + "\n")

    def test_integrality_gap(self):
        """
        Empirically measure the relaxation-vs-rounding gap across all available
        models and save two paper-ready figures:
          1) Top-K integrality gap vs scene complexity
          2) Boxplot of integrality gap by rounding method
        """
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
        if not os.path.exists(models_dir):
            self.skipTest("Models directory not found. Skipping integrality-gap test.")

        out_dir = os.path.join(os.path.dirname(__file__), 'test_out')
        os.makedirs(out_dir, exist_ok=True)

        scene_dirs = []
        for root, _, files in os.walk(models_dir):
            if 'point_cloud.ply' in files and 'iteration_' in root:
                scene_dirs.append(root)
        if not scene_dirs:
            self.skipTest("No model checkpoints found for integrality-gap test.")

        def parse_model_name(ply_root):
            parts = ply_root.split(os.sep)
            try:
                idx = parts.index('models')
                return parts[idx + 1]
            except Exception:
                return os.path.basename(os.path.dirname(os.path.dirname(ply_root)))

        def count_ply_vertices(ply_path):
            with open(ply_path, 'rb') as f:
                for raw_line in f:
                    line = raw_line.decode('utf-8', errors='ignore').strip()
                    if line.startswith('element vertex'):
                        return int(line.split()[-1])
                    if line == 'end_header':
                        break
            return None

        summary = []
        rounding_gaps = {
            'topK': [],
            'swap': [],
            'randomized': [],
            'pipage': [],
        }

        K = 10
        NUM_SPLATS = 300
        NUM_CAMERAS = 24

        for scene_root in sorted(scene_dirs):
            ply_path = os.path.join(scene_root, 'point_cloud.ply')
            scene_name = parse_model_name(scene_root)
            scene_complexity = count_ply_vertices(ply_path)

            with redirect_stdout(io.StringIO()):
                theta_params, _ = load_ply(ply_path, max_splats=NUM_SPLATS, align=True, flip=True)
                theta_params = normalize_theta_params(theta_params)
                cameras = generate_cameras_numpy(num_cameras=NUM_CAMERAS, radius=0.5)
                F_blocks = compute_fisher_information_numpy(theta_params, cameras)
                frob_norms = np.sqrt(np.einsum('mnab, mnab -> mn', F_blocks, F_blocks))
                lam = 1e-2 * np.mean(frob_norms[frob_norms > 0]) if np.any(frob_norms > 0) else 1e-2
                w_star, _ = solve_d_optimal_frank_wolfe_numpy(F_blocks, K=K, max_iter=60, lambda_reg=lam)

            relaxed_obj = continuous_relaxed_objective(F_blocks, w_star, lam)

            idx_topk = round_topK(w_star, K)
            idx_swap = round_swap_local_search(w_star, K, F_blocks, lam)
            idx_rand = round_randomized_best_of_N(w_star, K, F_blocks, lam, N_samples=50)
            idx_pipage = round_pipage(w_star, K, F_blocks, lam)

            discrete_objs = {
                'topK': discrete_subset_objective(F_blocks, idx_topk, lam),
                'swap': discrete_subset_objective(F_blocks, idx_swap, lam),
                'randomized': discrete_subset_objective(F_blocks, idx_rand, lam),
                'pipage': discrete_subset_objective(F_blocks, idx_pipage, lam),
            }
            gaps = {name: integrality_gap_percent(relaxed_obj, obj) for name, obj in discrete_objs.items()}
            for name, gap in gaps.items():
                rounding_gaps[name].append(gap)

            summary.append({
                'scene': scene_name,
                'scene_complexity': int(scene_complexity) if scene_complexity is not None else int(theta_params.shape[0]),
                'sampled_splats': int(theta_params.shape[0]),
                'topK_gap_pct': gaps['topK'],
                'swap_gap_pct': gaps['swap'],
                'randomized_gap_pct': gaps['randomized'],
                'pipage_gap_pct': gaps['pipage'],
                'relaxed_obj': relaxed_obj,
                'topK_discrete_obj': discrete_objs['topK'],
            })

        complexities = np.array([row['scene_complexity'] for row in summary], dtype=float)
        topk_gaps = np.array([row['topK_gap_pct'] for row in summary], dtype=float)
        corr = float(np.corrcoef(complexities, topk_gaps)[0, 1]) if len(summary) >= 2 and np.std(complexities) > 0 else None

        payload = {
            'config': {
                'K': K,
                'num_splats': NUM_SPLATS,
                'num_cameras': NUM_CAMERAS,
            },
            'pearson_corr_topK_gap_vs_scene_complexity': corr,
            'summary': summary,
        }
        with open(os.path.join(out_dir, 'integrality_gap_summary.json'), 'w') as f:
            json.dump(payload, f, indent=2)

        if summary:
            order = np.argsort(complexities)
            xs = complexities[order]
            ys = topk_gaps[order]
            labels = [summary[i]['scene'] for i in order]

            plt.figure(figsize=(7.5, 4.8), dpi=180)
            plt.scatter(xs, ys)
            for x, y, label in zip(xs, ys, labels):
                plt.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords='offset points')
            if len(xs) >= 2 and np.std(xs) > 0:
                coeffs = np.polyfit(xs, ys, 1)
                xfit = np.linspace(xs.min(), xs.max(), 100)
                yfit = coeffs[0] * xfit + coeffs[1]
                plt.plot(xfit, yfit, linestyle='--')
            plt.xlabel('Scene complexity (original splat count)')
            plt.ylabel('Integrality gap (%)')
            title = 'Integrality gap vs scene size (Top-K rounding)'
            if corr is not None:
                title += f'\nPearson r = {corr:.3f}'
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'integrality_gap_vs_scene_size.png'))
            plt.close()

        non_empty = [name for name, vals in rounding_gaps.items() if vals]
        if non_empty:
            plt.figure(figsize=(7.8, 4.8), dpi=180)
            plt.boxplot([rounding_gaps[name] for name in non_empty], labels=non_empty, showmeans=True)
            plt.ylabel('Integrality gap (%)')
            plt.title('Integrality gap by rounding method')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'integrality_gap_boxplot_by_rounding.png'))
            plt.close()

        for row in summary:
            self.assertGreaterEqual(row['topK_gap_pct'], -1e-5)
            self.assertLess(row['topK_gap_pct'], 100.0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
