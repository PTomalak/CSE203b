import numpy as np
import time
import struct
import os
import json

# 1/sigma^2: inverse noise variance on projected geometric quantities.
# The Fisher information is F = (1/sigma^2) * J^T J  (see Eq. 1 in the paper).
INV_SIGMA_SQ = 1e-4

def generate_cameras_numpy(num_cameras=100, radius=4.0):
    z_vals = np.linspace(1.0, 0.1, num_cameras)
    phi = np.arccos(z_vals)
    theta = np.pi * (1 + 5**0.5) * np.arange(num_cameras)
    
    cx = radius * np.sin(phi) * np.cos(theta)
    cy = radius * np.sin(phi) * np.sin(theta)
    cz = radius * np.cos(phi)
    
    camera_positions = np.stack([cx, cy, cz], axis=1)
    cameras = []
    
    for pos in camera_positions:
        forward = -pos / np.linalg.norm(pos)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-4:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        
        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)
        
        R = np.stack([right, down, forward])
        t = -R @ pos
        
        cameras.append({
            'R': R, 't': t, 'pos': pos,
            'fx': 800.0, 'fy': 800.0, 'cx': 400.0, 'cy': 400.0
        })
    return cameras

def read_colmap_cameras_bin(path):
    """
    Reads COLMAP cameras.bin and returns dict of camera_id → intrinsics.
    Handles PINHOLE model (fx, fy, cx, cy).
    """
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id  = struct.unpack("<I", f.read(4))[0]
            width     = struct.unpack("<Q", f.read(8))[0]
            height    = struct.unpack("<Q", f.read(8))[0]
            
            num_params = {0: 3, 1: 4, 2: 4, 3: 5}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            
            if model_id == 0:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            
            cameras[camera_id] = {
                "fx": fx, "fy": fy,
                "cx": cx, "cy": cy,
                "width": width, "height": height
            }
    return cameras

def read_colmap_images_bin(path):
    """
    Reads COLMAP images.bin and returns list of dicts with R, t, name.
    """
    cameras = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)
            
            R = quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            
            cameras.append({
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
                "R": R,
                "t": t,
            })
    return cameras

def quat_to_rotmat(qw, qx, qy, qz):
    """Unit quaternion (w,x,y,z) → 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R

def load_real_cameras(scene_dir):
    """
    Load real cameras from either:
      1. COLMAP sparse/0/{images.bin,cameras.bin}
      2. cameras.json fallback
    """
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    cameras_json = os.path.join(scene_dir, "cameras.json")

    if os.path.exists(images_bin) and os.path.exists(cameras_bin):
        images = read_colmap_images_bin(images_bin)
        cam_map = read_colmap_cameras_bin(cameras_bin)
        result = []
        for img in images:
            intr = cam_map[img["camera_id"]]
            result.append({
                "R": img["R"],
                "t": img["t"],
                "fx": intr["fx"],
                "fy": intr["fy"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "name": img["name"],
            })
        return result

    if os.path.exists(cameras_json):
        return load_cameras_json(cameras_json)

    raise FileNotFoundError(
        f"No camera metadata found in {scene_dir}. "
        f"Expected either sparse/0/images.bin + cameras.bin or cameras.json"
    )

def project_gaussian_batched(theta_N, R_c, t_c, fx, fy, cx, cy):
    mu = theta_N[:, 0:3]
    scale = theta_N[:, 3:6]
    quat = theta_N[:, 6:10]
    
    q_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    q = quat / q_norm
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R_00 = 1 - 2*(y**2 + z**2)
    R_01 = 2*(x*y - r*z)
    R_02 = 2*(x*z + r*y)
    R_10 = 2*(x*y + r*z)
    R_11 = 1 - 2*(x**2 + z**2)
    R_12 = 2*(y*z - r*x)
    R_20 = 2*(x*z - r*y)
    R_21 = 2*(y*z + r*x)
    R_22 = 1 - 2*(x**2 + y**2)
    
    R = np.stack([
        np.stack([R_00, R_01, R_02], axis=1),
        np.stack([R_10, R_11, R_12], axis=1),
        np.stack([R_20, R_21, R_22], axis=1)
    ], axis=1)
    
    S_diag = np.exp(scale)
    RS = R * S_diag[:, np.newaxis, :]
    Sigma_3d = RS @ np.transpose(RS, axes=(0, 2, 1))
    
    t_pt = (R_c @ mu.T).T + t_c
    tx = t_pt[:, 0]
    ty = t_pt[:, 1]
    tz = t_pt[:, 2]
    
    px = fx * (tx / tz) + cx
    py = fy * (ty / tz) + cy
    mu_2d = np.stack([px, py], axis=1)
    
    J1_0 = fx / tz
    J1_1 = np.zeros_like(tz)
    J1_2 = -fx * tx / (tz**2)
    J2_0 = np.zeros_like(tz)
    J2_1 = fy / tz
    J2_2 = -fy * ty / (tz**2)
    
    J_proj = np.stack([
        np.stack([J1_0, J1_1, J1_2], axis=1),
        np.stack([J2_0, J2_1, J2_2], axis=1)
    ], axis=1)
    
    T1 = J_proj @ R_c
    Sigma_2d = T1 @ Sigma_3d @ np.transpose(T1, axes=(0, 2, 1))
    
    sig_00 = Sigma_2d[:, 0, 0] + 0.3
    sig_11 = Sigma_2d[:, 1, 1] + 0.3
    sig_01 = Sigma_2d[:, 0, 1]
    
    return np.stack([mu_2d[:, 0], mu_2d[:, 1], sig_00, sig_01, sig_11], axis=1)


def estimate_output_scales(theta_params, cameras, n_sample_cams=10):
    """
    Estimate the standard deviation of each of the 5 projection outputs
    across a sample of cameras and all primitives.
    
    Returns a (5,) vector of scales. Dividing the Jacobian rows by these
    scales ensures that J^T J treats position and covariance information
    equally, rather than being dominated by pixel-coordinate derivatives.
    """
    N = theta_params.shape[0]
    sample_cams = cameras[:min(n_sample_cams, len(cameras))]
    
    all_outputs = []
    for cam in sample_cams:
        R_c, t_c = cam['R'], cam['t']
        fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
        f_out = project_gaussian_batched(theta_params, R_c, t_c, fx, fy, cx, cy)
        all_outputs.append(f_out)
    
    all_outputs = np.concatenate(all_outputs, axis=0) 
    
    scales = np.std(all_outputs, axis=0)
    scales = np.maximum(scales, 1e-6)
    
    return scales

def compute_fisher_information_numpy(theta_params, cameras, progress_callback=None,
                                     normalize_outputs=True):
    """
    Compute Fisher information blocks F_{j,i} = J̃^T J̃ for all views j and primitives i.
    
    When normalize_outputs=True (default, NEW), the Jacobian rows are divided by
    the empirical std of each output dimension. This ensures that position and
    covariance terms contribute equally to the Fisher information, rather than
    pixel-coordinate derivatives dominating by factors of 10^3+.
    
    Parameters
    ----------
    theta_params : ndarray (N, 10)
    cameras : list of dict
    progress_callback : callable or None
    normalize_outputs : bool — if True, normalize Jacobian by output scales
    
    Returns
    -------
    F_blocks : ndarray (M, N, 10, 10)
    """
    N = theta_params.shape[0]
    M = len(cameras)
    
    # Estimate normalization weights
    if normalize_outputs:
        output_scales = estimate_output_scales(theta_params, cameras)
        W = 1.0 / output_scales
    else:
        W = np.ones(5)
    
    F_blocks = np.zeros((M, N, 10, 10))
    eps = 1e-4
    
    for j, cam in enumerate(cameras):
        if progress_callback:
            progress_callback((j + 1) / M, f"Computing Jacobians: Camera {j+1}/{M}")
            
        R_c, t_c = cam['R'], cam['t']
        fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
        
        J_all = np.zeros((N, 5, 10))
        
        for p in range(10):
            theta_plus = theta_params.copy()
            theta_plus[:, p] += eps
            f_plus = project_gaussian_batched(theta_plus, R_c, t_c, fx, fy, cx, cy)
            
            theta_minus = theta_params.copy()
            theta_minus[:, p] -= eps
            f_minus = project_gaussian_batched(theta_minus, R_c, t_c, fx, fy, cx, cy)
            
            J_all[:, :, p] = (f_plus - f_minus) / (2 * eps)
        
        # J_normalized[n, k, p] = J[n, k, p] * W[k]
        J_all = J_all * W[np.newaxis, :, np.newaxis]
        
        # Fisher block: F_{j,i} = J̃^T J̃
        F = np.transpose(J_all, axes=(0, 2, 1)) @ J_all
        
        t_pt = (R_c @ theta_params[:, 0:3].T).T + t_c
        valid_mask = t_pt[:, 2] > 0.2
        
        F_scaled = F * INV_SIGMA_SQ
        F_blocks[j, valid_mask] = np.nan_to_num(F_scaled[valid_mask])
                
    return F_blocks

def _fw_line_search(M_w_base, s_star, w, F_blocks, I_expand):
    """
    Exact line search: γ* = argmin_{γ∈[0,1]} f((1-γ)w + γ s*)
    
    Uses the identity: M(w + γd) = M(w) + γ·D  where D = Σ_j d_j F_{j,i}
    so only one einsum is needed (to compute D), then each γ evaluation
    is just M_w_base + γ*D — a cheap matrix addition + slogdet.
    
    Parameters
    ----------
    M_w_base : ndarray (N, 10, 10) — precomputed M(w) = Σ_j w_j F_j + λI
    s_star : ndarray (M,) — LMO solution
    w : ndarray (M,) — current weights
    F_blocks : ndarray (M, N, 10, 10)
    I_expand : ndarray (1, 10, 10) — λI (for broadcasting)
    """
    d = s_star - w
    D = np.einsum('m, mnab -> nab', d, F_blocks)
    
    def objective_at_gamma(gamma):
        M_gamma = M_w_base + gamma * D
        return -np.sum(np.linalg.slogdet(M_gamma)[1])
    
    a, b = 0.0, 1.0
    gr = (np.sqrt(5) + 1) / 2
    tol = 1e-4  
    c = b - (b - a) / gr
    d_pt = a + (b - a) / gr
    fc, fd = objective_at_gamma(c), objective_at_gamma(d_pt)
    
    for _ in range(20):
        if fc < fd:
            b = d_pt
            d_pt, fd = c, fc
            c = b - (b - a) / gr
            fc = objective_at_gamma(c)
        else:
            a = c
            c, fc = d_pt, fd
            d_pt = a + (b - a) / gr
            fd = objective_at_gamma(d_pt)
        if b - a < tol:
            break
    
    return (a + b) / 2


def solve_d_optimal_frank_wolfe_numpy(F_blocks, K, max_iter=200, lambda_reg=None,
                                       lambda_frac=1e-2, progress_callback=None,
                                       use_line_search=True):
    """
    Solve the D-optimal view selection via Frank-Wolfe.
    
    Uses exact line search instead of fixed γ=2/(t+2) step size.
    This guarantees monotonic objective decrease and much tighter convergence.
    
    Parameters
    ----------
    F_blocks : ndarray (M, N, 10, 10)
    K : int — budget
    max_iter : int
    lambda_reg : float or None
    lambda_frac : float
    use_line_search : bool — if True, use golden-section line search (recommended)
    
    Returns
    -------
    w : ndarray (M,)
    history_gap : list of float
    """
    M, N, _, _ = F_blocks.shape
    
    if lambda_reg is None:
        frob_norms = np.sqrt(np.einsum('mnab, mnab -> mn', F_blocks, F_blocks))
        mean_norm = np.mean(frob_norms[frob_norms > 0]) if np.any(frob_norms > 0) else 1.0
        lambda_reg = lambda_frac * mean_norm
    
    w = np.ones(M) * (K / M)
    I_N = np.eye(10)
    I_expand = lambda_reg * np.expand_dims(I_N, 0) 
    
    history_gap = []
    
    for t in range(max_iter):
        M_w = np.einsum('m, mnab -> nab', w, F_blocks) + I_expand
        M_inv = np.linalg.inv(M_w)
        
        grads = -np.einsum('nab, mnba -> m', M_inv, F_blocks)
        
        top_indices = np.argsort(grads)[:K]
        s_star = np.zeros(M)
        s_star[top_indices] = 1.0
        
        gap = np.dot(grads, w - s_star)
        history_gap.append(gap)
        
        if use_line_search:
            gamma = _fw_line_search(M_w, s_star, w, F_blocks, I_expand)
        else:
            gamma = 2.0 / (t + 2)
        
        w = (1 - gamma) * w + gamma * s_star
        
        if progress_callback:
            progress_callback((t+1)/max_iter, f"Iter {t+1:03d} | Gap: {gap:.6f} | γ={gamma:.4f}")
        
        obj = -np.sum(np.linalg.slogdet(M_w)[1])
        rel_gap = gap / max(abs(obj), 1e-10)
        if rel_gap < 1e-6 and t > 5:
            break
            
    if progress_callback:
        progress_callback(1.0, f"Converged in {t+1} iters | Gap: {gap:.6f}")
    return w, history_gap



def continuous_relaxed_objective(F_blocks, w, lambda_reg):
    """Return the minimization objective -log det(M(w))."""
    I_expand = lambda_reg * np.eye(10)
    M_w = np.einsum('m, mnab -> nab', w, F_blocks) + I_expand
    return float(-np.sum(np.linalg.slogdet(M_w)[1]))


def discrete_subset_objective(F_blocks, indices, lambda_reg):
    """Return the discrete minimization objective -log det(M(S))."""
    I_expand = lambda_reg * np.eye(10)
    M_w = F_blocks[indices].sum(axis=0) + I_expand
    return float(-np.sum(np.linalg.slogdet(M_w)[1]))


def integrality_gap_percent(relaxed_obj, discrete_obj):
    """
    Empirical integrality-gap percentage for the minimization objective.

    Since the convex relaxation is a lower bound, this should be >= 0 up to
    numerical tolerance.
    """
    denom = max(abs(relaxed_obj), 1e-12)
    return 100.0 * (discrete_obj - relaxed_obj) / denom

def round_topK(w, K):
    """Baseline: pick K cameras with largest w_j."""
    return np.argsort(w)[-K:].tolist()


def round_swap_local_search(w, K, F_blocks, lambda_reg, max_rounds=50):
    """
    Swap-based local search after top-K rounding.
    
    Start from top-K, then greedily swap one camera in/out if it
    improves the discrete D-optimal objective. Repeat until no
    improving swap exists.
    
    This is the single most impactful fix for closing the rounding gap.
    """
    M, N, _, _ = F_blocks.shape
    I_expand = lambda_reg * np.eye(10)
    
    # Start from top-K
    selected = set(np.argsort(w)[-K:].tolist())
    
    def discrete_obj(sel):
        """Compute -log det M(S) for a discrete set S."""
        idx = list(sel)
        M_w = F_blocks[idx].sum(axis=0) + I_expand  # (N, 10, 10)
        return -np.sum(np.linalg.slogdet(M_w)[1])
    
    current_obj = discrete_obj(selected)
    
    for round_num in range(max_rounds):
        improved = False
        
        for j_in in range(M):
            if j_in in selected:
                continue
            
            for j_out in selected:
                # Try swapping j_out for j_in
                candidate = (selected - {j_out}) | {j_in}
                new_obj = discrete_obj(candidate)
                
                if new_obj < current_obj - 1e-10:  # strict improvement
                    selected = candidate
                    current_obj = new_obj
                    improved = True
                    break  # restart inner loop with new set
            
            if improved:
                break
        
        if not improved:
            break
    
    return sorted(selected)


def round_randomized_best_of_N(w, K, F_blocks, lambda_reg, N_samples=100):
    """
    Randomized rounding — sample K cameras proportional to w,
    repeat N_samples times, keep the best.
    
    This preserves the distributional information from the relaxation
    that top-K throws away.
    """
    M = len(w)
    I_expand = lambda_reg * np.eye(10)
    
    probs = np.maximum(w, 0)
    probs = probs / probs.sum()  # normalize to probability distribution
    
    best_obj = np.inf
    best_idx = None
    
    for _ in range(N_samples):
        # Sample K cameras without replacement, proportional to w
        idx = np.random.choice(M, size=K, replace=False, p=probs)
        M_w = F_blocks[idx].sum(axis=0) + I_expand
        obj = -np.sum(np.linalg.slogdet(M_w)[1])
        if obj < best_obj:
            best_obj = obj
            best_idx = idx.tolist()
    
    return best_idx


def round_pipage(w, K, F_blocks, lambda_reg):
    """
    Pipage rounding for matroid-constrained submodular maximization.
    
    Iteratively makes fractional entries more integral while maintaining
    or improving the objective. Guaranteed to produce an integral solution
    with objective >= the fractional optimum (for monotone submodular).
    
    Reference: Calinescu et al., "Maximizing a Monotone Submodular Function
    Subject to a Matroid Constraint", SICOMP 2011.
    """
    M, N, _, _ = F_blocks.shape
    I_expand = lambda_reg * np.eye(10)
    
    w_curr = w.copy()
    
    def obj_at(ww):
        M_w = np.einsum('m, mnab -> nab', ww, F_blocks) + I_expand[np.newaxis]
        return -np.sum(np.linalg.slogdet(M_w)[1])
    
    max_iters = M * 10
    for iteration in range(max_iters):
        # Find two fractional entries
        frac_indices = np.where((w_curr > 1e-10) & (w_curr < 1 - 1e-10))[0]
        
        if len(frac_indices) < 2:
            break
        
        i, j = frac_indices[0], frac_indices[1]
        
        # Increase w_i, decrease w_j (or vice versa)
        eps_up_i   = min(1.0 - w_curr[i], w_curr[j])       # push i toward 1, j toward 0
        eps_down_i = min(w_curr[i], 1.0 - w_curr[j])       # push i toward 0, j toward 1
        
        if eps_up_i < 1e-12 and eps_down_i < 1e-12:
            continue
        
        # Evaluate objective at both extremes
        w_up = w_curr.copy()
        w_up[i] += eps_up_i
        w_up[j] -= eps_up_i
        obj_up = obj_at(w_up)
        
        w_down = w_curr.copy()
        w_down[i] -= eps_down_i
        w_down[j] += eps_down_i
        obj_down = obj_at(w_down)
        
        if obj_up <= obj_down:
            w_curr = w_up
        else:
            w_curr = w_down
    
    # Snap remaining fractional entries
    selected = np.where(w_curr > 0.5)[0].tolist()
    
    # If we got more or fewer than K due to numerical issues, adjust
    if len(selected) > K:
        # Remove the ones with smallest w
        excess = len(selected) - K
        sub_w = [(w_curr[s], s) for s in selected]
        sub_w.sort()
        for _, s in sub_w[:excess]:
            selected.remove(s)
    elif len(selected) < K:
        # Add the fractional ones with largest w
        remaining = [i for i in range(M) if i not in selected]
        remaining.sort(key=lambda i: -w_curr[i])
        selected.extend(remaining[:K - len(selected)])
    
    return sorted(selected[:K])

def solve_and_round(F_blocks, K, rounding='swap', max_iter=200, lambda_reg=None,
                    lambda_frac=1e-2, progress_callback=None):
    """
    Convenience function: solve the relaxation, then round.
    
    Parameters
    ----------
    rounding : str — 'topK', 'swap', 'randomized', 'pipage'
    
    Returns
    -------
    indices : list of int — K selected camera indices
    w_star : ndarray (M,) — continuous relaxation weights
    history : list — duality gap history
    """
    M, N, _, _ = F_blocks.shape
    
    # Compute shared lambda
    if lambda_reg is None:
        frob_norms = np.sqrt(np.einsum('mnab, mnab -> mn', F_blocks, F_blocks))
        mean_norm = np.mean(frob_norms[frob_norms > 0]) if np.any(frob_norms > 0) else 1.0
        lambda_reg = lambda_frac * mean_norm
    
    w_star, history = solve_d_optimal_frank_wolfe_numpy(
        F_blocks, K, max_iter=max_iter, lambda_reg=lambda_reg,
        progress_callback=progress_callback)
    
    if rounding == 'topK':
        indices = round_topK(w_star, K)
    elif rounding == 'swap':
        indices = round_swap_local_search(w_star, K, F_blocks, lambda_reg)
    elif rounding == 'randomized':
        indices = round_randomized_best_of_N(w_star, K, F_blocks, lambda_reg)
    elif rounding == 'pipage':
        indices = round_pipage(w_star, K, F_blocks, lambda_reg)
    else:
        raise ValueError(f"Unknown rounding: {rounding}")
    
    return indices, w_star, history, lambda_reg

def load_cameras_json(path):
    """
    Load cameras from a Gaussian Splatting-style cameras.json file.

    Expected keys typically include:
      - rotation or R
      - position or T / translation
      - fx, fy, cx, cy
    """
    with open(path, "r") as f:
        data = json.load(f)

    cameras = []

    for cam in data:
        # Rotation
        if "rotation" in cam:
            R = np.array(cam["rotation"], dtype=float)
        elif "R" in cam:
            R = np.array(cam["R"], dtype=float)
        else:
            raise ValueError(f"Camera entry missing rotation/R keys: {cam.keys()}")

        # Translation / position handling
        if "position" in cam:
            pos = np.array(cam["position"], dtype=float)
            t = -R @ pos
        elif "t" in cam:
            t = np.array(cam["t"], dtype=float)
        elif "translation" in cam:
            t = np.array(cam["translation"], dtype=float)
        elif "T" in cam:
            t = np.array(cam["T"], dtype=float)
        else:
            raise ValueError(f"Camera entry missing position/t/translation/T keys: {cam.keys()}")

        fx = float(cam.get("fx", 800.0))
        fy = float(cam.get("fy", 800.0))
        cx = float(cam.get("cx", 400.0))
        cy = float(cam.get("cy", 400.0))

        cameras.append({
            "R": R,
            "t": t,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "name": cam.get("img_name", cam.get("image_name", "unknown")),
        })

    return cameras