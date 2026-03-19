"""
run_experiment.py — Benchmark D-optimal view selection with all fixes.

Methods tested:
  - random (averaged over 10 draws)
  - uniform (evenly spaced)
  - greedy_nbv (sequential greedy, FisherRF/OUGS baseline)
  - doptimal_topK (FW relaxation + naive top-K rounding)
  - doptimal_swap (FW relaxation + swap local search)          ← NEW
  - doptimal_randomized (FW relaxation + randomized rounding)  ← NEW
  - doptimal_pipage (FW relaxation + pipage rounding)          ← NEW

Outputs:
  results.json — all results, convergence data, weight vectors
"""

import numpy as np
import time
import json
import os
from src.loader import load_ply, normalize_theta_params
from src.fast_solver import (
    generate_cameras_numpy,
    compute_fisher_information_numpy,
    solve_d_optimal_frank_wolfe_numpy,
    load_real_cameras,
    round_topK,
    round_swap_local_search,
    round_randomized_best_of_N,
    round_pipage,
)

SCENES = {
    "truck":     "data/tandt_db/tandt/truck",
    "train":     "data/tandt_db/tandt/train",
    "drjohnson": "data/tandt_db/db/drjohnson",
    "playroom":  "data/tandt_db/db/playroom",
    "flowers":   "data/360_extra_scenes/flowers",
    "treehill":  "data/360_extra_scenes/treehill",
    "bicycle":   "data/360_v2/bicycle",
    "bonsai":    "data/360_v2/bonsai",
    "counter":   "data/360_v2/counter",
    "garden":    "data/360_v2/garden",
    "kitchen":   "data/360_v2/kitchen",
    "room":      "data/360_v2/room",
    "stump":     "data/360_v2/stump",
}
PLY_ROOT   = "src/models"
NUM_SPLATS = 1000
K_VALUES   = [5, 10, 20]
REPEATS    = 10

METHODS = [
    "random",
    "uniform",
    "greedy_nbv",
    "doptimal_topK",
    "doptimal_swap",
    "doptimal_randomized",
    "doptimal_pipage",
]


# ── baselines ────────────────────────────────────────────────────────────────

def select_random(M, K):
    return np.random.choice(M, K, replace=False).tolist()

def select_uniform(M, K):
    step = max(1, M // K)
    return list(range(0, M, step))[:K]

def select_greedy_nbv(F_blocks, K, lam):
    """Greedy NBV with shared lambda."""
    M, N, _, _ = F_blocks.shape
    selected = []
    M_inv = (1.0 / lam) * np.eye(10)[np.newaxis].repeat(N, axis=0)
    for k in range(K):
        best_j, best_gain = -1, -np.inf
        for j in range(M):
            if j in selected:
                continue
            inner = np.eye(10) + np.einsum('nab,nbc->nac', M_inv, F_blocks[j])
            gain = np.sum(np.linalg.slogdet(inner)[1])
            if gain > best_gain:
                best_gain, best_j = gain, j
        selected.append(best_j)
        F_j = F_blocks[best_j]
        MiF = np.einsum('nab,nbc->nac', M_inv, F_j)
        inner = np.eye(10) + MiF
        inner_inv = np.linalg.inv(inner)
        M_inv = M_inv - np.einsum('nab,nbc,ncd->nad', MiF, inner_inv, M_inv)
    return selected


# ── scoring ──────────────────────────────────────────────────────────────────

def logdet_score(F_blocks, indices, lam):
    """D-optimal logdet with FIXED lambda."""
    F_subset = F_blocks[indices]
    M_w = F_subset.sum(axis=0) + lam * np.eye(10)
    return float(np.sum(np.linalg.slogdet(M_w)[1]))

def compute_shared_lambda(F_blocks, frac=1e-2):
    frob_norms = np.sqrt(np.einsum('mnab, mnab -> mn', F_blocks, F_blocks))
    mean_norm = np.mean(frob_norms[frob_norms > 0]) if np.any(frob_norms > 0) else 1.0
    return frac * mean_norm


# ── main ─────────────────────────────────────────────────────────────────────

results = {}
scene_meta = {}

for scene_name, scene_dir in SCENES.items():
    ply_path = os.path.join(PLY_ROOT, scene_name,
                            "point_cloud/iteration_30000/point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"[SKIP] {scene_name} — .ply not found at {ply_path}")
        continue

    print(f"\n{'='*70}\n  Scene: {scene_name}\n{'='*70}")

    np.random.seed(42)
    theta, _ = load_ply(ply_path, max_splats=NUM_SPLATS, align=False, flip=False)
    cameras  = load_real_cameras(scene_dir)

    # Scale theta to camera coordinate range
    cam_positions = np.array([-c['R'].T @ c['t'] for c in cameras])
    cam_scale   = np.percentile(np.abs(cam_positions), 95)
    theta_scale = np.percentile(np.abs(theta[:, :3]), 95)
    ratio = cam_scale / theta_scale
    theta[:, 0:3] *= ratio
    theta[:, 3:6] += np.log(ratio)

    M = len(cameras)
    print(f"  {M} real cameras, {theta.shape[0]} splats")

    # Compute Fisher (with normalized outputs)
    print("  Computing Fisher blocks (normalized outputs)...")
    t0 = time.time()
    F_blocks = compute_fisher_information_numpy(theta, cameras, normalize_outputs=True)
    fisher_time = time.time() - t0
    print(f"  Fisher done in {fisher_time:.1f}s")

    shared_lam = compute_shared_lambda(F_blocks)
    print(f"  Shared λ = {shared_lam:.6e}")

    scene_meta[scene_name] = {
        "num_cameras": M,
        "num_splats": theta.shape[0],
        "fisher_time": fisher_time,
        "shared_lambda": shared_lam,
    }

    fw_cache = {}
    for K in K_VALUES:
        print(f"\n  ── K={K} ──")
        t0 = time.time()
        w_star, gap_history = solve_d_optimal_frank_wolfe_numpy(
            F_blocks, K=K, max_iter=300, lambda_reg=shared_lam,
            use_line_search=True)
        fw_time = time.time() - t0
        fw_cache[K] = (w_star, gap_history, fw_time)
        print(f"  FW solved in {fw_time:.2f}s ({len(gap_history)} iters)")

        for method in METHODS:
            key = f"{scene_name}_{method}_K{K}"

            if method == "random":
                scores, times, all_idx = [], [], []
                for _ in range(REPEATS):
                    t0 = time.time()
                    idx = select_random(M, K)
                    times.append(time.time() - t0)
                    scores.append(logdet_score(F_blocks, idx, shared_lam))
                    all_idx.append(idx)
                best_run = int(np.argmax(scores))
                entry = {
                    "scene": scene_name, "method": method, "K": K,
                    "logdet": float(np.mean(scores)),
                    "logdet_std": float(np.std(scores)),
                    "solve_time": float(np.mean(times)),
                    "selected": all_idx[best_run],
                    "fw_iters": None,
                }

            elif method == "uniform":
                t0 = time.time()
                idx = select_uniform(M, K)
                solve_t = time.time() - t0
                entry = {
                    "scene": scene_name, "method": method, "K": K,
                    "logdet": logdet_score(F_blocks, idx, shared_lam),
                    "logdet_std": 0.0,
                    "solve_time": solve_t,
                    "selected": idx,
                    "fw_iters": None,
                }

            elif method == "greedy_nbv":
                t0 = time.time()
                idx = select_greedy_nbv(F_blocks, K, shared_lam)
                solve_t = time.time() - t0
                entry = {
                    "scene": scene_name, "method": method, "K": K,
                    "logdet": logdet_score(F_blocks, idx, shared_lam),
                    "logdet_std": 0.0,
                    "solve_time": solve_t,
                    "selected": idx,
                    "fw_iters": None,
                }

            elif method.startswith("doptimal_"):
                w_star, gap_history, fw_time_K = fw_cache[K]
                rounding_name = method.split("_", 1)[1]

                t0 = time.time()
                if rounding_name == "topK":
                    idx = round_topK(w_star, K)
                elif rounding_name == "swap":
                    idx = round_swap_local_search(w_star, K, F_blocks, shared_lam)
                elif rounding_name == "randomized":
                    idx = round_randomized_best_of_N(w_star, K, F_blocks, shared_lam, N_samples=200)
                elif rounding_name == "pipage":
                    idx = round_pipage(w_star, K, F_blocks, shared_lam)
                else:
                    raise ValueError(f"Unknown rounding: {rounding_name}")
                round_time = time.time() - t0

                # Continuous relaxation objective
                N_splats = F_blocks.shape[1]
                M_w_cont = np.einsum('m, mnab -> nab', w_star, F_blocks) \
                         + shared_lam * np.expand_dims(np.eye(10), 0)
                relaxed_obj = float(-np.sum(np.linalg.slogdet(M_w_cont)[1]))

                score = logdet_score(F_blocks, idx, shared_lam)
                entry = {
                    "scene": scene_name, "method": method, "K": K,
                    "logdet": score,
                    "logdet_std": 0.0,
                    "solve_time": fw_time_K + round_time,
                    "round_time": round_time,
                    "selected": idx,
                    "fw_iters": len(gap_history),
                    "relaxed_obj": relaxed_obj,
                    "discrete_obj": float(-score),
                }
                if rounding_name == "topK":
                    # Store convergence data and weights once
                    entry["fw_gap_history"] = [float(g) for g in gap_history]
                    entry["w_star"] = [float(w) for w in w_star]

            results[key] = entry
            tag = "→" if "doptimal" in method else " "
            print(f"    {tag} {method:<22} logdet={entry['logdet']:12.3f}"
                  f" (±{entry['logdet_std']:.1f})  t={entry['solve_time']:.3f}s"
                  f"  sel={entry['selected'][:5]}...")


# ── save ─────────────────────────────────────────────────────────────────────

output = {
    "config": {
        "ply_root": PLY_ROOT, "num_splats": NUM_SPLATS,
        "k_values": K_VALUES, "methods": METHODS, "repeats": REPEATS,
        "normalize_outputs": True, "use_line_search": True,
    },
    "scene_meta": scene_meta,
    "results": results,
}

with open("results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved results.json ({len(results)} entries across {len(scene_meta)} scenes)")