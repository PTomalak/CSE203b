import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rich.console import Console
from rich.table import Table

from loader import load_ply, normalize_theta_params
from fast_solver import generate_cameras_numpy, compute_fisher_information_numpy, solve_d_optimal_frank_wolfe_numpy

console = Console()


def visualize_results(theta_params, cameras, w, K, save_path, colors=None):
    """
    Create 3D visualization of the D-optimal view selection results.

    Parameters
    ----------
    theta_params : ndarray (N, 10) - geometric parameters
    cameras : list of dict - camera dictionaries with 'pos' key (ndarray)
    w : ndarray (M,) - optimal weights
    K : int - number of views to highlight
    save_path : str - where to save the figure
    colors : ndarray (N, 3) or None - RGB colors in [0,1] for splats
    """
    console.print("\nVisualizing Results...")

    top_indices = np.argsort(w)[::-1][:K]

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    mu = theta_params[:, 0:3]
    idx = np.random.choice(mu.shape[0], min(2000, mu.shape[0]), replace=False)

    # Use provided colors or default to blue
    if colors is not None:
        splat_colors = colors[idx]
        # Ensure colors are in [0,1]
        if splat_colors.max() > 1.0:
            splat_colors = splat_colors / 255.0
        splat_colors = np.clip(splat_colors, 0, 1)
    else:
        splat_colors = 'blue'

    ax.scatter(mu[idx, 0], mu[idx, 1], mu[idx, 2], c=splat_colors, s=1.0, alpha=0.3, label='3D Gaussians (Sampled)')

    cam_pos = np.stack([c['pos'] for c in cameras])
    ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], c='grey', s=20, alpha=0.4, label='Candidate Views')

    selected_pos = cam_pos[top_indices]
    ax.scatter(selected_pos[:, 0], selected_pos[:, 1], selected_pos[:, 2], c='red', s=300, marker='*', edgecolor='black', linewidth=0.5, label=f'Selected Top {K} Views')

    for pos in selected_pos:
        ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], c='red', alpha=0.3, linestyle='--')

    ax.set_title("D-Optimal View Selection (Block-Diagonal Frank-Wolfe)")

    ax.view_init(elev=30, azim=45)

    max_range = np.array([cam_pos[:,0].max()-cam_pos[:,0].min(), cam_pos[:,1].max()-cam_pos[:,1].min(), cam_pos[:,2].max()-cam_pos[:,2].min()]).max() / 2.0
    mid_x = (cam_pos[:,0].max()+cam_pos[:,0].min()) * 0.5
    mid_y = (cam_pos[:,1].max()+cam_pos[:,1].min()) * 0.5
    mid_z = (cam_pos[:,2].max()+cam_pos[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    plt.tight_layout()
    plt.savefig(save_path)
    console.print(f"[bold green]Saved plot to[/bold green] {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to pre-trained 3DGS .ply file")
    parser.add_argument("--num_splats", type=int, default=1000, help="Downsample splats for rapid testing")
    parser.add_argument("--num_cameras", type=int, default=200, help="Number of candidate cameras on sphere")
    parser.add_argument("--budget", type=int, default=10, help="Budget (K) of chosen views")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius of camera sphere (default: 0.5, range: 0.1-2)")
    parser.add_argument("--lambda_frac", type=float, default=1e-2, help="Fraction of mean Fisher norm used for adaptive regularization")
    parser.add_argument("--align", action='store_true', help="Automatically align object to Z-up using PCA (default: True)")
    parser.add_argument("--no-align", dest='align', action='store_false', help="Disable automatic alignment")
    parser.add_argument("--flip", action='store_true', help="Apply heuristic Z-flip after PCA alignment (default: True)")
    parser.add_argument("--no-flip", dest='flip', action='store_false', help="Disable heuristic Z-flip")
    parser.add_argument("--blue", action='store_true', help="Force blue color for splats (ignores PLY colors)")
    parser.add_argument("--verify", action='store_true', help="Run dense matrix verification (slow, memory-intensive; only for small N)")
    parser.set_defaults(align=True, flip=True, verify=False)
    args = parser.parse_args()

    np.random.seed(42)

    # 1. Load Data
    theta_params, colors_from_ply = load_ply(args.ply, max_splats=args.num_splats, align=args.align, flip=args.flip)
    # Override colors with blue if requested
    colors = None if args.blue else colors_from_ply
    theta_params = normalize_theta_params(theta_params)

    cameras = generate_cameras_numpy(num_cameras=args.num_cameras, radius=args.radius)
    def log_cb(pct, msg): print(f"  {msg}", end="\r", flush=True)
    F_blocks = compute_fisher_information_numpy(theta_params, cameras, progress_callback=log_cb)
    print()  # Newline after iterative loop
    w_star, history = solve_d_optimal_frank_wolfe_numpy(F_blocks, args.budget, max_iter=200, lambda_frac=args.lambda_frac, progress_callback=lambda p, m: print(f"  {m}", end="\r", flush=True))
    print()

    # 4. Mathematically Verify the trace in Numpy (optional)
    if args.verify:
        console.print("\n[bold]Running Mathematical Verification (Numpy Dense Inverse Check)[/bold]")
        I_N = np.eye(10)

        # Recompute the same adaptive lambda used by the solver
        frob_norms = np.sqrt(np.einsum('mnab, mnab -> mn', F_blocks, F_blocks))
        lam = args.lambda_frac * np.mean(frob_norms[frob_norms > 0]) if np.any(frob_norms > 0) else args.lambda_frac

        M_w = np.einsum('m, mnab -> nab', w_star, F_blocks) + lam * np.expand_dims(I_N, 0).repeat(theta_params.shape[0], axis=0)
        block_inv = np.linalg.inv(M_w)

        trace_block = np.einsum('nii->', block_inv)

        dense_M = np.zeros((10*theta_params.shape[0], 10*theta_params.shape[0]))
        for i in range(theta_params.shape[0]):
            dense_M[10*i:10*(i+1), 10*i:10*(i+1)] = M_w[i]

        dense_inv = np.linalg.inv(dense_M)
        trace_dense = np.trace(dense_inv)

        table = Table(title="Trace Verification Results")
        table.add_column("Property", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("Block Diagonal Inverse Shape", str(block_inv.shape))
        table.add_row("Sum of Traces (Block)", f"{trace_block:.4f}")
        table.add_row("Trace of Dense Inverse", f"{trace_dense:.4f}")
        match_str = "[bold green]True[/bold green]" if abs(trace_block - trace_dense) < 1e-3 else "[bold red]False[/bold red]"
        table.add_row("Match Exactly?", match_str)

        console.print(table)

    # 5. Visualize
    visualize_results(theta_params, cameras, w_star, args.budget, "/home/ek/syncthing/fast_sync/203b_proj/CSE_203B(1)/src/verification.png", colors=colors)
