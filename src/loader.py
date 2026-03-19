"""
Unified PLY loading utilities for 3D Gaussian Splatting models.

Provides both file-based and URL-based loading with consistent
alignment, normalization, and color extraction logic.
"""

import numpy as np
from src.align import align_object_to_z_up


def parse_ply_vertices(vertex_data):
    """
    Core parsing logic shared across all loaders.

    Parameters
    ----------
    vertex_data : structured array or dict-like
        PLY vertex data with fields: x, y, z, scale_0-2, rot_0-3,
        optionally f_dc_0-2 or red/green/blue

    Returns
    -------
    theta_params : ndarray (N, 10) - [mu, scale, quat]
    colors : ndarray (N, 3) or None - RGB in [0, 1]
    """
    # Extract positions
    x = vertex_data['x'].copy()
    y = vertex_data['y'].copy()
    z = vertex_data['z'].copy()
    mu = np.stack([x, y, z], axis=1).astype(np.float32)

    # Extract scales (log-scale)
    scale = np.stack([
        vertex_data['scale_0'].copy(),
        vertex_data['scale_1'].copy(),
        vertex_data['scale_2'].copy()
    ], axis=1).astype(np.float32)

    # Extract quaternions (w, x, y, z) and normalize
    quat = np.stack([
        vertex_data['rot_0'].copy(),
        vertex_data['rot_1'].copy(),
        vertex_data['rot_2'].copy(),
        vertex_data['rot_3'].copy()
    ], axis=1).astype(np.float32)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    # Extract colors (DC SH coefficients or direct RGB)
    colors = None
    if all(k in vertex_data for k in ['f_dc_0', 'f_dc_1', 'f_dc_2']):
        r = vertex_data['f_dc_0'].copy()
        g = vertex_data['f_dc_1'].copy()
        b = vertex_data['f_dc_2'].copy()
        colors = np.stack([r, g, b], axis=1).astype(np.float32)
        colors = np.clip(colors, 0, 1)
    elif all(k in vertex_data for k in ['red', 'green', 'blue']):
        r = vertex_data['red'].copy()
        g = vertex_data['green'].copy()
        b = vertex_data['blue'].copy()
        colors = np.stack([r, g, b], axis=1).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0, 1)

    # Concatenate into parameter vector
    theta_params = np.concatenate([mu, scale, quat], axis=1)

    return theta_params, colors


def load_ply_from_file(ply_path, max_splats=None, align=True, flip=True):
    """
    Load 3DGS .ply file from local filesystem.

    Parameters
    ----------
    ply_path : str
        Path to .ply file
    max_splats : int or None
        Randomly downsample to this many splats (for testing)
    align : bool
        Apply PCA-based Z-up alignment
    flip : bool
        Apply heuristic Z-flip after alignment

    Returns
    -------
    theta_params : ndarray (N, 10)
    colors : ndarray (N, 3) or None
    """
    from rich.console import Console
    from plyfile import PlyData
    console = Console()

    console.print(f"[bold blue]Loading PLY file:[/bold blue] {ply_path}")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']

    theta_params, colors = parse_ply_vertices(v)

    if max_splats is not None and max_splats < theta_params.shape[0]:
        console.print(f"Downsampling from {theta_params.shape[0]} to {max_splats} splats for faster testing.")
        idx = np.random.permutation(theta_params.shape[0])[:max_splats]
        theta_params = theta_params[idx]
        if colors is not None:
            colors = colors[idx]

    if align:
        theta_params = align_object_to_z_up(theta_params, flip=flip)

    console.print(f"[green]Loaded {theta_params.shape[0]} Gaussians[/green] with 10 geometric parameters.")
    if colors is not None:
        console.print(f"[cyan]Loaded colors with shape {colors.shape}[/cyan]")
    else:
        console.print("[yellow]No colors found in PLY file; will use default blue[/yellow]")

    return theta_params, colors


def normalize_theta_params(theta_params):
    """
    Center geometry at origin and scale to unit sphere.
    Adjusts log-scales accordingly.

    Parameters
    ----------
    theta_params : ndarray (N, 10)

    Returns
    -------
    theta_params : ndarray (N, 10) - normalized in-place
    """
    xyz = theta_params[:, 0:3]
    center = np.mean(xyz, axis=0)
    theta_params[:, 0:3] -= center

    max_dist = np.max(np.linalg.norm(theta_params[:, 0:3], axis=1))
    if max_dist > 1e-4:
        scale_factor = 1.0 / max_dist
        theta_params[:, 0:3] *= scale_factor
        theta_params[:, 3:6] += np.log(scale_factor)

    return theta_params


def _get_vertex_data_from_file(ply_path):
    """Internal: Load vertex data from local file."""
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    return plydata['vertex']


def _get_vertex_data_from_url(url, max_splats, progress_callback):
    """Internal: Fetch vertex data from URL via HTTP Range."""
    import urllib.request
    import re

    if progress_callback:
        progress_callback(0, "Fetching PLY headers...")

    try:
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-4095'})
        resp = urllib.request.urlopen(req)
        header_data = resp.read()
    except Exception as e:
        import traceback
        raise RuntimeError(f"Failed to fetch {url}: {e}\n{traceback.format_exc()}")

    header_end = header_data.find(b'end_header\n') + 11
    match = re.search(b'element vertex (\\d+)', header_data)
    if not match:
        raise ValueError("Invalid PLY file format: could not find element vertex count")

    total_splats = int(match.group(1))
    bytes_per_splat = 248
    fetch_cnt = min(max_splats, total_splats) if max_splats else total_splats

    if progress_callback:
        progress_callback(0.1, f"Model has {total_splats} splats ({(total_splats * 248) / 1024**2:.1f} MB). Fetching {fetch_cnt} samples...")

    blocks = min(5, fetch_cnt)
    points_per_block = fetch_cnt // blocks
    all_vertex_data = []

    for i in range(blocks):
        start_splat = i * (total_splats // blocks)
        start_byte = header_end + start_splat * bytes_per_splat
        end_byte = start_byte + points_per_block * bytes_per_splat - 1

        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}'})
            resp = urllib.request.urlopen(req)
            if resp.status == 200:
                raise RuntimeError("Server ignored HTTP Range; tried to send whole file. Server must support Range requests.")
            data = resp.read()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch chunk {i}: {e}")

        block_data = np.frombuffer(data, dtype=np.float32).reshape(-1, 62)
        all_vertex_data.append(block_data)

        if progress_callback:
            progress_callback(0.1 + 0.8 * (i + 1) / blocks, f"Downloaded block {i+1}/{blocks}")

    vertex_data = np.concatenate(all_vertex_data, axis=0)

    # Create a dict-like object for parse_ply_vertices
    class VertexData:
        def __init__(self, data):
            self.data = data
            self.field_indices = {
                'x': 0, 'y': 1, 'z': 2,
                'scale_0': 55, 'scale_1': 56, 'scale_2': 57,
                'rot_0': 58, 'rot_1': 59, 'rot_2': 60, 'rot_3': 61,
                'f_dc_0': 6, 'f_dc_1': 7, 'f_dc_2': 8,
                'red': 6, 'green': 7, 'blue': 8
            }
        def __getitem__(self, key):
            idx = self.field_indices[key]
            return self.data[:, idx]
        def __contains__(self, key):
            return key in self.field_indices

    return VertexData(vertex_data)


def load_ply(source, max_splats=None, align=True, flip=True, progress_callback=None):
    """
    Unified PLY loading interface. Auto-detects source type (file path vs URL).

    Parameters
    ----------
    source : str
        File path or URL (http/https) to .ply file
    max_splats : int or None
        Downsample to this many splats (for testing)
    align : bool
        Apply PCA-based Z-up alignment
    flip : bool
        Apply heuristic Z-flip after alignment
    progress_callback : callable(pct, msg) or None
        Progress reporting (used for URL fetches; files are silent)

    Returns
    -------
    theta_params : ndarray (N, 10)
    colors : ndarray (N, 3) or None
    """
    # Auto-detect if source is a URL
    is_url = source.startswith('http://') or source.startswith('https://')

    # Get vertex data from appropriate source
    if is_url:
        v = _get_vertex_data_from_url(source, max_splats, progress_callback)
    else:
        v = _get_vertex_data_from_file(source)

    # Parse vertices
    theta_params, colors = parse_ply_vertices(v)

    # Downsample if requested
    if max_splats is not None and max_splats < theta_params.shape[0]:
        idx = np.random.permutation(theta_params.shape[0])[:max_splats]
        theta_params = theta_params[idx]
        if colors is not None:
            colors = colors[idx]

    # Align if requested
    if align:
        theta_params = align_object_to_z_up(theta_params, flip=flip)

    # Print summary (only for file-based loading with console)
    if not is_url:
        from rich.console import Console
        console = Console()
        console.print(f"[green]Loaded {theta_params.shape[0]} Gaussians[/green] with 10 geometric parameters.")
        if colors is not None:
            console.print(f"[cyan]Loaded colors with shape {colors.shape}[/cyan]")
        else:
            console.print("[yellow]No colors found in PLY file; will use default blue[/yellow]")

    return theta_params, colors


# Backwards compatibility: keep old function names as aliases
def load_ply_from_file(source, **kwargs):
    """Explicit file loader (source is a file path)."""
    return load_ply(source, **kwargs)

def fetch_ply_from_url(source, **kwargs):
    """Explicit URL loader (source is an http/https URL)."""
    return load_ply(source, **kwargs)
