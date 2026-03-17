import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go

from loader import load_ply, normalize_theta_params
from fast_solver import generate_cameras_numpy, compute_fisher_information_numpy, solve_d_optimal_frank_wolfe_numpy

st.set_page_config(layout="wide", page_title="D-Optimal Active View Selection")

# ==========================================
# Models & Data Loading
# ==========================================

st.title("Geometric Active View Selection")
st.markdown("""
This demo runs **entirely inside your browser** using WebAssembly (including all required python libraries)
Select a 3D Gaussian Splatting model below.
The app will fetch the specific `.ply` model point cloud directly to your RAM and use pure Numpy + Autograd to analytically project and solve the Frank-Wolfe D-Optimal discrete selection budget.
""")

# List of all available models and their sizes (in MB)
AVAILABLE_MODELS = {
    "train": 242.8,
    "counter": 289.2,
    "bonsai": 294.4,
    "room": 376.9,
    "kitchen": 438.1,
    "truck": 601.0,
    "playroom": 602.2,
    "drjohnson": 805.4,
    "flowers": 860.1,
    "treehill": 894.9,
    "stump": 1173.5,
    "garden": 1379.0,
    "bicycle": 1450.3
}
# Sort models by size for default selection
AVAILABLE_MODELS = dict(sorted(AVAILABLE_MODELS.items(), key=lambda item: item[1]))

col_settings, col_viz = st.columns([1, 2])

with col_settings:
    st.header("1. Selection Settings")
    model_name_display = st.selectbox(
        "Select 3DGS Scene Model", 
        list(AVAILABLE_MODELS.keys()), 
        format_func=lambda x: f"{x} ({AVAILABLE_MODELS[x]:.1f} MB)",
        index=0
    )
    # The actual string model name
    model_name = model_name_display
    
    num_splats = st.slider("Max Splats to Extract Feature Bounds", min_value=100, max_value=6000, value=500, step=100)
    num_cameras = st.slider("Candidate Camera Sphere Resolution", min_value=20, max_value=300, value=100, step=20)
    budget = st.slider("Camera View Budget (K)", min_value=5, max_value=50, value=10, step=1)
    radius = st.slider("Camera Sphere Radius", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                       help="Radius of the camera hemisphere. Smaller values (0.1-0.5) are better for detailed objects, larger values (1-2) for larger scenes.")
    
    current_params = {
        "model": model_name,
        "splats": num_splats,
        "cameras": num_cameras,
        "budget": budget,
        "radius": radius
    }
    
    params_changed = st.session_state.get('run_params') is not None and st.session_state.get('run_params') != current_params
    
    if params_changed:
        st.markdown(
            """
            <style>
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
                100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
            }
            .stButton button[kind="primary"] {
                animation: pulse 1.5s infinite;
                border: 2px solid #ff4b4b;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.warning("Parameters changed! Rerun to update")
        
    force_blue = st.checkbox("Force blue color (ignore PLY colors)", value=False,
                             help="Render all splats as blue instead of using colors from the .ply file")
    auto_align = st.checkbox("Auto-align to Z-up (PCA)", value=True,
                             help="Automatically rotate object so its 'up' direction aligns with camera hemisphere Z-axis")
    auto_flip = st.checkbox("Apply Z-flip heuristic", value=True,
                            help="After PCA alignment, flip object if it appears upside down (median Z > 0)")
    
    if st.button("Run Optimizer", type="primary"):
        run_optimization = True
    else:
        run_optimization = False

# ==========================================
# Math & Solver Functions
# ==========================================

def load_ply_numpy(model_name, max_splats, align=True, flip=True):
    """
    Load and optionally auto-align a 3DGS model from the local server.
    Uses unified load_ply() from loader.py with auto-detected URL.
    
    Parameters
    ----------
    model_name : str
    max_splats : int or None
    align : bool - apply PCA-based Z-up alignment
    flip : bool - apply heuristic Z-flip after alignment
    
    Returns
    -------
    theta_params : ndarray (N, 10)
    colors : ndarray (N, 3) or None
    """
    if max_splats is None:
        max_splats = 500

    url = f"http://localhost:8000/models/{model_name}/point_cloud/iteration_30000/point_cloud.ply"
    
    theta_params, colors = load_ply(
        url,
        max_splats=max_splats,
        align=align,
        flip=flip,
    )
    
    return theta_params, colors

# ==========================================
# Execution Main Thread
# ==========================================

if run_optimization:
    st.markdown(
        """
        <style>
        @keyframes computing-pulse {
            0% { opacity: 0.6; box-shadow: 0 0 0 0 rgba(150, 150, 150, 0.7); }
            50% { opacity: 1.0; box-shadow: 0 0 0 10px rgba(150, 150, 150, 0); }
            100% { opacity: 0.6; box-shadow: 0 0 0 0 rgba(150, 150, 150, 0); }
        }
        .stButton button[kind="primary"] {
            pointer-events: none !important;
            animation: computing-pulse 1.5s infinite !important;
            background-color: #888888 !important;
            border-color: #888888 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.spinner("Executing Pipeline... (UI will freeze during WebAssembly computation)"):
        
        with col_settings:
            st.header("2. Pipeline Status")
            
            # 1. Load Data
            theta_params, colors_from_ply = load_ply_numpy(model_name, max_splats=num_splats, align=auto_align, flip=auto_flip)
            colors = None if force_blue else colors_from_ply
            theta_params = normalize_theta_params(theta_params)
            cameras = generate_cameras_numpy(num_cameras=num_cameras, radius=radius)
            
            # 2. Extract Fisher Info via fast_solver
            t0 = time.time()
            F_blocks = compute_fisher_information_numpy(theta_params, cameras)
            st.success(f"Extracted {len(cameras) * len(theta_params)} Jacobians via pure Vectorized Numpy in {time.time()-t0:.2f}s")
            
            # 3. Solve FW Trace Minimization
            t1 = time.time()
            w_star, history = solve_d_optimal_frank_wolfe_numpy(F_blocks, budget)
            st.success(f"Frank-Wolfe Converged strictly fulfilling duality bounds! (Time: {time.time()-t1:.2f}s)")
            
    with col_viz:
        st.header("3. Optimal 3D View Plot")
        
        top_indices = w_star.argsort()[::-1][:budget]

        # Draw 3D Plotly Map
        mu = theta_params[:, 0:3]
        cam_pos = np.stack([c['pos'] for c in cameras])
        selected_pos = cam_pos[top_indices]
        
        fig = go.Figure()
        
        # Add Splats with colors from PLY file
        NEON_BLUE = 'rgb(0, 191, 255)'  # Deep sky blue
        splat_colors = colors if colors is not None else NEON_BLUE
        # If colors are in [0,1], convert to 0-255 for Plotly
        if isinstance(splat_colors, np.ndarray) and splat_colors.max() <= 1.0:
            splat_colors = (splat_colors * 255).astype(np.uint8)
            # Convert to RGB tuples for Plotly
            color_tuples = [f'rgb({r},{g},{b})' for r,g,b in splat_colors]
        elif isinstance(splat_colors, np.ndarray):
            color_tuples = [f'rgb({r},{g},{b})' for r,g,b in splat_colors]
        else:
            color_tuples = splat_colors
            
        fig.add_trace(go.Scatter3d(
            x=mu[:, 0], y=mu[:, 1], z=mu[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_tuples if isinstance(splat_colors, np.ndarray) else splat_colors,
                opacity=0.3
            ),
            name='Gaussian Splats'
        ))
        
        # Add Candidate Cameras
        fig.add_trace(go.Scatter3d(
            x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
            mode='markers',
            marker=dict(size=4, color='rgba(200, 200, 200, 0.4)'),
            name='Candidate Views'
        ))
        
        # Add Selected Optimal Cameras
        fig.add_trace(go.Scatter3d(
            x=selected_pos[:, 0], y=selected_pos[:, 1], z=selected_pos[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', symbol='cross'),
            name='Selected D-Optimal Views'
        ))
        
        # Formulate ray lines
        for pos in selected_pos:
            fig.add_trace(go.Scatter3d(
                x=[0, pos[0]], y=[0, pos[1]], z=[0, pos[2]],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ))
            
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(50,50,50,0.5)", font=dict(color="white")),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            height=700
        )
        st.session_state['fig'] = fig
        st.session_state['run_params'] = current_params

if 'fig' in st.session_state:
    with col_viz:
        if not run_optimization:
            st.header("3. Optimal 3D View Plot")
        st.plotly_chart(st.session_state['fig'], use_container_width=True)
elif not run_optimization:
    with col_viz:
        st.info("← Adjust the settings and click **Run Optimizer** to execute the pipeline")
