# Geometric Active View Selection: D-Optimal Approach

Pure NumPy implementation of convex D-optimal active view selection for 3D Gaussian Splatting. Optimizes for geometric accuracy rather than photometric quality.

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib plyfile rich streamlit plotly

# Download pretrained 3DGS models
cd src && python download_models.py

# Run local test (CPU, ~30-90s)
python local.py --ply models/train/point_cloud/iteration_30000/point_cloud.ply --num_splats 500 --budget 10 --radius 0.2

# Or launch interactive web demo
python serve.py  # Terminal static file server on port 8000
# open http://localhost:8000/demo.html in browser
```

## Project Structure

```
src/
├── fast_solver.py      # Core: Fisher extraction + Frank-Wolfe solver
├── loader.py           # Unified PLY loading (file/URL, alignment, downsampling)
├── align.py            # PCA inspired Z-up alignment utility
├── local.py            # Local execution with matplotlib output
├── demo.py             # Browser-based Streamlit demo (WebAssembly)
├── download_models.py  # Download pretrained models from INRIA
├── serve.py            # Simple HTTP server for demo
├── test_pipeline.py    # Unit tests and robustness validation
└── models/             # Pretrained 3DGS scenes (after download_models.py)
```

## Core Components

### `fast_solver.py`

- `generate_cameras_numpy(num_cameras, radius)`: Fibonacci hemisphere sampling
- `project_gaussian_batched(theta, R, t, fx, fy, cx, cy)`: EWA projection (5D output)
- `compute_fisher_information_numpy(theta, cameras)`: Central FD Jacobian (ε=1e-4), returns (M, N, 10, 10) blocks
- `solve_d_optimal_frank_wolfe_numpy(F_blocks, K, lambda_frac=1e-2)`: Frank-Wolfe with adaptive λ, returns optimal weights

### `loader.py`

- `load_ply(source, max_splats, align, flip)`: Unified API for file paths and URLs. Handles downsampling, PCA alignment, Z-flip heuristic, and normalization. Returns `(theta_params, colors)`.

### `align.py`

- `align_object_to_z_up(theta_params, flip)`: PCA-based rotation to align object's principal axis with +Z. Optional heuristic flip ensures base points downward. (Requires further improvement for some scenes but covers most cases)

## Execution Options

### 1. Local Script (`local.py`)

Full pipeline with console output and matplotlib visualization.

```bash
python local.py \
  --ply /path/to/model.ply \
  --num_splats 1000 \
  --num_cameras 200 \
  --budget 10 \
  --radius 0.5 \
  --lambda_frac 0.01 \
  [--align] [--no-align] \
  [--flip] [--no-flip] \
  [--blue] [--verify]
```

**Parameters:**
- `--ply`: Path to `.ply` file (required)
- `--num_splats`: Downsample limit for testing (default: 1000)
- `--num_cameras`: Candidate views on hemisphere (default: 200)
- `--budget`: Number of views to select (K) (default: 10)
- `--radius`: Camera sphere radius, range [0.1, 2.0] (default: 0.5)
- `--lambda_frac`: Adaptive regularization fraction (default: 0.01)
- `--align` / `--no-align`: Enable/disable PCA Z-up alignment (default: enabled)
- `--flip` / `--no-flip`: Apply heuristic Z-flip after alignment (default: enabled)
- `--blue`: Force blue splats, ignore PLY colors
- `--verify`: Run dense matrix verification (slow, memory-intensive; only for small N)

**Output:** Saves `src/verification.png` with 3D plot.

### 2. Web Demo (`demo.py`)

Interactive browser-based demo running entirely in WebAssembly via stlite.

**Prerequisites:**
1. `pip install streamlit plotly`
2. Models downloaded to `src/models/`
3. Static file server running on port 8000 (see below)

**Start server:**
```bash
cd src
python serve.py  # Serves models/ via HTTP with Range request support
```

**Launch demo:**
```bash
# open http://localhost:8000/demo.html in browser
```

The demo:
- Fetches PLY headers to get splat count
- Downloads sparse chunks via HTTP Range
- Runs optimization entirely in-browser (NumPy + WebAssembly)
- Displays interactive Plotly 3D visualization

**Controls:**
- Scene selector (13 pretrained models)
- Max splats (100–6000)
- Candidate cameras (20–300)
- Budget K (5–50)
- Camera radius (0.1–2.0)
- Alignment/flip/color options

### 3. Tests (`test_pipeline.py`)

Unit tests and robustness sweep across all downloaded models.

```bash
python test_pipeline.py
```

**Test coverage:**
- Camera generation: hemisphere placement, orthonormality
- Fisher blocks: shape (M, N, 10, 10) and positive semi-definiteness
- Frank-Wolfe: constraints (sum(w)=K, 0≤w≤1), block-diagonal trace equivalence
- Real models: sweep 4 configurations across all 13 scenes, generate plots in `src/test_out/`

## Implementation Notes

- **Pure NumPy**: No PyTorch, no CUDA. All gradients via central finite differences.
- **Block-diagonal structure**: Fisher matrix is block-diagonal with n independent 10×10 blocks. Inversion complexity O(n) instead of O(n³).
- **Adaptive λ**: λ = 1e-2 × mean Frobenius norm of non-zero Fisher blocks. Scales with scene magnitude.
- **Memory**: Fisher blocks scale as O(M × N × 10 × 10). For N=1000, M=200, expect ~160 MB.
- **Runtime**: Finite differences dominate (~N×M×10 projections). N=1000, M=200 → ~10–30s on laptop CPU.

## File Format

Input: Standard 3DGS `.ply` with vertex fields:
- Positions: `x`, `y`, `z`
- Scales: `scale_0`, `scale_1`, `scale_2` (log-scale)
- Orientations: `rot_0`, `rot_1`, `rot_2`, `rot_3` (quaternion w,x,y,z)
- Colors (optional): either SH DC coefficients `f_dc_0`, `f_dc_1`, `f_dc_2` or RGB `red`, `green`, `blue`

## References

- Paper: *Geometric Active View Selection: A Convex D-Optimal Approach* (CSE 203B, UC San Diego)
- Related: FisherRF (ECCV 2024), OUGS (arXiv 2025), POp-GS (CVPR 2025)
- 3DGS: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
