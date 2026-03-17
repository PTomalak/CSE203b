import numpy as np
import time

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

def project_gaussian_batched(theta_N, R_c, t_c, fx, fy, cx, cy):
    # theta_N: (N, 10)
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
    
    # J_proj: (N, 2, 3), R_c: (3, 3)
    T1 = J_proj @ R_c
    Sigma_2d = T1 @ Sigma_3d @ np.transpose(T1, axes=(0, 2, 1))
    
    sig_00 = Sigma_2d[:, 0, 0] + 0.3
    sig_11 = Sigma_2d[:, 1, 1] + 0.3
    sig_01 = Sigma_2d[:, 0, 1]
    
    return np.stack([mu_2d[:, 0], mu_2d[:, 1], sig_00, sig_01, sig_11], axis=1)

def compute_fisher_information_numpy(theta_params, cameras, progress_callback=None):
    """
    Compute Fisher information blocks F_{j,i} = J_{g,j,i}^T J_{g,j,i} for all views j and primitives i.
    
    Uses central finite differences to compute the geometric Jacobian:
        J_{g,j,i}[k,:] = (f(θ + ε e_k) - f(θ - ε e_k)) / (2ε)
    where f = [μ̃, vech(Σ̃)] is the EWA projection output (5-dimensional).
    
    Parameters
    ----------
    theta_params : ndarray (N, 10) - geometric parameters for N primitives
    cameras : list of dict - camera extrinsics and intrinsics
    progress_callback : callable - optional progress reporting
    
    Returns
    -------
    F_blocks : ndarray (M, N, 10, 10) - Fisher blocks F_{j,i} for each view j, primitive i
    """
    N = theta_params.shape[0]
    M = len(cameras)
    
    F_blocks = np.zeros((M, N, 10, 10))
    t0 = time.time()
    eps = 1e-4  # Finite difference step size (Eq. 9)
    
    for j, cam in enumerate(cameras):
        if progress_callback:
            progress_callback((j + 1) / M, f"Computing Jacobians (Numpy FD): Camera {j+1}/{M}")
            
        R_c, t_c = cam['R'], cam['t']
        fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
        
        # Base projection f(θ) for all N primitives
        f_base = project_gaussian_batched(theta_params, R_c, t_c, fx, fy, cx, cy)
        J_all = np.zeros((N, 5, 10))
        
        # Central finite differences for each of the 10 parameter dimensions
        for p in range(10):
            theta_shifted_plus = theta_params.copy()
            theta_shifted_plus[:, p] += eps
            f_plus = project_gaussian_batched(theta_shifted_plus, R_c, t_c, fx, fy, cx, cy)
            
            theta_shifted_minus = theta_params.copy()
            theta_shifted_minus[:, p] -= eps
            f_minus = project_gaussian_batched(theta_shifted_minus, R_c, t_c, fx, fy, cx, cy)
            
            # J[:,:,p] = ∂f/∂θ_p for all primitives simultaneously
            J_all[:, :, p] = (f_plus - f_minus) / (2 * eps)
            
        # Fisher block: F_{j,i} = J_{g,j,i}^T J_{g,j,i} (Eq. 1)
        F = np.transpose(J_all, axes=(0, 2, 1)) @ J_all
        
        t_pt = (R_c @ theta_params[:, 0:3].T).T + t_c
        valid_mask = t_pt[:, 2] > 0.2
        
        F_scaled = F * INV_SIGMA_SQ
        F_blocks[j, valid_mask] = np.nan_to_num(F_scaled[valid_mask])
                
    return F_blocks

def solve_d_optimal_frank_wolfe_numpy(F_blocks, K, max_iter=100, lambda_reg=None, lambda_frac=1e-2, progress_callback=None):
    """
    Solve the D-optimal view selection problem via Frank-Wolfe.
    
    Implements Algorithm 1 from the paper: "Geometric Active View Selection: A Convex D-Optimal Approach"
    
    Parameters
    ----------
    F_blocks : ndarray (M, N, 10, 10) - per-view, per-primitive Fisher blocks F_{j,i}
    K : int - budget of views to select
    max_iter : int - maximum FW iterations
    lambda_reg : float or None - explicit regularization. If None, computed
                 adaptively as lambda_frac * mean Frobenius norm of F_blocks (Eq. 10).
    lambda_frac : float - fraction of mean Fisher norm used for adaptive regularization (default 1e-2).
    
    Returns
    -------
    w : ndarray (M,) - optimal weights w* (sum to K, each in [0,1])
    history_gap : list - duality gap history for convergence monitoring
    
    Notes
    -----
    The Frank-Wolfe algorithm (Section 4.1) iteratively:
    1. Compute gradient: ∇f(w)_j = -∑_{i=1}^n tr(M_i(w)^{-1} F_{j,i}) (Eq. 11)
    2. Solve LMO: s* = argmin_{s∈C} ⟨∇f(w), s⟩ by picking K most negative gradients (Eq. 12)
    3. Update: w ← (1-γ)w + γ s* with γ = 2/(t+2) (Eq. 13)
    """
    M, N, _, _ = F_blocks.shape
    
    # Adaptive regularization (Eq. 10): λ = α · (1/|S|) ∑_{(j,i)∈S} ||F_{j,i}||_F
    # where S = {(j,i) : F_{j,i} ≠ 0} and α = lambda_frac (default 1e-2)
    if lambda_reg is None:
        # Compute Frobenius norm of each (view, primitive) block: ||F_{j,i}||_F
        frob_norms = np.sqrt(np.einsum('mnab, mnab -> mn', F_blocks, F_blocks))  # (M, N)
        # Average over non-zero blocks only
        mean_norm = np.mean(frob_norms[frob_norms > 0]) if np.any(frob_norms > 0) else 1.0
        lambda_reg = lambda_frac * mean_norm
    
    w = np.ones(M) * (K / M)
    I_N = np.eye(10)
    
    history_gap = []
    
    for t in range(max_iter):
        # M(w) = ∑_j w_j F_j + λ I (block-diagonal, Eq. 10)
        M_w = np.einsum('m, mnab -> nab', w, F_blocks) + lambda_reg * np.expand_dims(I_N, 0)
        # Invert each 10×10 block separately: M_i(w)^{-1} for i=1..n (Section 4.1)
        M_inv = np.linalg.inv(M_w)
        
        # Gradient: ∇f(w)_j = -∑_{i=1}^n tr(M_i(w)^{-1} F_{j,i}) (Eq. 11)
        # The einsum computes: for each view j, sum over all primitives i: trace(M_inv[i] @ F_blocks[j,i])
        grads = -np.einsum('nab, mnba -> m', M_inv, F_blocks)
        
        # Linear Minimization Oracle (LMO): s* = argmin_{s∈C} ⟨∇f(w), s⟩ (Eq. 12)
        # Over simplex C = {s | 1^T s = K, 0 ≤ s ≤ 1}, solved greedily by picking K most negative gradients
        top_indices = np.argsort(grads)[:K]
        s_star = np.zeros(M)
        s_star[top_indices] = 1.0  # Set selected views to full weight (K/K = 1)
        
        # Duality gap: g(w) = max_{s∈C} ⟨∇f(w), w-s⟩ = ⟨∇f(w), w-s*⟩ (Frank-Wolfe theory)
        gap = np.dot(grads, w - s_star)
        history_gap.append(gap)
        
        # Step size: γ_t = 2/(t+2) (Eq. 13, ensures convergence)
        gamma = 2.0 / (t + 2)
        w = (1 - gamma) * w + gamma * s_star
        
        if progress_callback:
            progress_callback((t+1)/max_iter, f"Iter {t+1:03d} | Gap: {gap:.6f} | λ={lambda_reg:.2e}")
        
        # Objective: f(w) = -log det M(w) = -∑_{i=1}^n log det M_i(w)
        # Using slogdet for numerical stability: sum of log(abs(det)) since M_i ≻ 0
        obj = -np.sum(np.linalg.slogdet(M_w)[1])
        
        # Relative duality gap: g_t / |f(w)| < ε (Eq. 14)
        rel_gap = gap / max(abs(obj), 1e-10)
        if rel_gap < 1e-6 and t > 5:
            break
            
    if progress_callback:
        progress_callback(1.0, f"Converged in {t+1} iters (Gap: {gap:.6f} | λ={lambda_reg:.2e})")
    return w, history_gap
