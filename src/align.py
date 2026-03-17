import numpy as np

def align_object_to_z_up(theta_params, flip=True):
    """
    Pure NumPy PCA-based Z-up alignment.
    Uses SVD of centered data to compute principal components and rotates
    the object so that the most appropriate axis (based on variance distribution) aligns with Z.
    The axis is chosen heuristically: if the largest variance is dominant, it is used (tall objects);
    if the smallest variance is distinct, it is used (flat objects); otherwise defaults to smallest.
    After alignment, an optional heuristic flip can be applied to ensure the object's base
    (denser part) is at negative Z.

    Parameters
    ----------
    theta_params : ndarray (N, 10) - geometric parameters for N primitives
    flip : bool - if True, apply heuristic flip if median Z > 0 (default: True)

    Returns
    -------
    aligned_params : ndarray (N, 10) - rotated parameters
    """
    xyz = theta_params[:, 0:3].copy()  # (N, 3)

    # Center the points
    centroid = xyz.mean(axis=0)
    xyz_centered = xyz - centroid

    # Compute SVD of centered data
    # U: (N, 3), S: (3,), Vt: (3, 3) - Vt contains principal components as rows
    U, S, Vt = np.linalg.svd(xyz_centered, full_matrices=False)
    principal_axes = Vt  # Each row is a principal direction

    # The singular values squared are proportional to variance
    # Adaptive axis selection: determine whether the object is "tall" (largest variance dominates)
    # or "flat" (smallest variance is distinct). Use ratios to decide.
    variances = S ** 2
    threshold = 2.0  # Tune this threshold as needed
    if variances[0] / variances[1] > threshold:
        # Largest variance is dominant -> object is elongated in one direction (tall)
        up_axis_idx = 0
    elif variances[1] / variances[2] > threshold:
        # Smallest variance is distinct -> object is flat (thin in one direction)
        up_axis_idx = 2
    else:
        # No clear dominance; default to smallest variance (flat assumption)
        up_axis_idx = 2
    up_direction = principal_axes[up_axis_idx]

    # Ensure up_direction points generally upward (positive Z)
    if up_direction[2] < 0:
        up_direction = -up_direction

    # Build rotation matrix to align up_direction to [0, 0, 1]
    z_axis = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(up_direction, z_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6:
        # Already aligned (or opposite, but we flipped above)
        R = np.eye(3) if np.dot(up_direction, z_axis) > 0 else np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_theta = np.dot(up_direction, z_axis)
        sin_theta = rotation_axis_norm
        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

    # Apply rotation to positions (and add back centroid if we want, but normalization will center anyway)
    xyz_aligned = xyz_centered @ R.T  # Apply rotation to centered data

    # Rotate the orientations (quaternions)
    quats = theta_params[:, 6:10].copy()
    quats_aligned = np.zeros_like(quats)
    for i in range(len(quats)):
        q = quats[i]
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        Q = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        Q_new = R @ Q
        # Convert back to quaternion using trace method
        tr = Q_new[0,0] + Q_new[1,1] + Q_new[2,2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w_new = 0.25 * S
            x_new = (Q_new[2,1] - Q_new[1,2]) / S
            y_new = (Q_new[0,2] - Q_new[2,0]) / S
            z_new = (Q_new[1,0] - Q_new[0,1]) / S
        else:
            if Q_new[0,0] > Q_new[1,1] and Q_new[0,0] > Q_new[2,2]:
                S = np.sqrt(1.0 + Q_new[0,0] - Q_new[1,1] - Q_new[2,2]) * 2
                w_new = (Q_new[2,1] - Q_new[1,2]) / S
                x_new = 0.25 * S
                y_new = (Q_new[0,1] + Q_new[1,0]) / S
                z_new = (Q_new[0,2] + Q_new[2,0]) / S
            elif Q_new[1,1] > Q_new[2,2]:
                S = np.sqrt(1.0 + Q_new[1,1] - Q_new[0,0] - Q_new[2,2]) * 2
                w_new = (Q_new[0,2] - Q_new[2,0]) / S
                x_new = (Q_new[0,1] + Q_new[1,0]) / S
                y_new = 0.25 * S
                z_new = (Q_new[1,2] + Q_new[2,1]) / S
            else:
                S = np.sqrt(1.0 + Q_new[2,2] - Q_new[0,0] - Q_new[1,1]) * 2
                w_new = (Q_new[1,0] - Q_new[0,1]) / S
                x_new = (Q_new[0,2] + Q_new[2,0]) / S
                y_new = (Q_new[1,2] + Q_new[2,1]) / S
                z_new = 0.25 * S
        quats_aligned[i] = [w_new, x_new, y_new, z_new]

    # Reconstruct aligned parameters (positions will be re-centered later by normalization)
    aligned_params = theta_params.copy()
    aligned_params[:, 0:3] = xyz_aligned + centroid  # Add back centroid before normalization
    aligned_params[:, 6:10] = quats_aligned

    # Heuristic flip: if median Z > 0, object is likely upside down, flip it (only if flip=True)
    if flip:
        median_z = np.median(xyz_aligned[:, 2])
        if median_z > 0:
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            # Flip positions
            xyz_flipped = xyz_aligned @ R_flip.T
            aligned_params[:, 0:3] = xyz_flipped + centroid
            # Flip quaternions
            quats_flipped = np.zeros_like(quats_aligned)
            for i in range(len(quats_aligned)):
                q = quats_aligned[i]
                q = q / np.linalg.norm(q)
                w, x, y, z = q
                Q = np.array([
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
                ])
                Q_new = R_flip @ Q
                tr = Q_new[0,0] + Q_new[1,1] + Q_new[2,2]
                if tr > 0:
                    S = np.sqrt(tr + 1.0) * 2
                    w_new = 0.25 * S
                    x_new = (Q_new[2,1] - Q_new[1,2]) / S
                    y_new = (Q_new[0,2] - Q_new[2,0]) / S
                    z_new = (Q_new[1,0] - Q_new[0,1]) / S
                else:
                    if Q_new[0,0] > Q_new[1,1] and Q_new[0,0] > Q_new[2,2]:
                        S = np.sqrt(1.0 + Q_new[0,0] - Q_new[1,1] - Q_new[2,2]) * 2
                        w_new = (Q_new[2,1] - Q_new[1,2]) / S
                        x_new = 0.25 * S
                        y_new = (Q_new[0,1] + Q_new[1,0]) / S
                        z_new = (Q_new[0,2] + Q_new[2,0]) / S
                    elif Q_new[1,1] > Q_new[2,2]:
                        S = np.sqrt(1.0 + Q_new[1,1] - Q_new[0,0] - Q_new[2,2]) * 2
                        w_new = (Q_new[0,2] - Q_new[2,0]) / S
                        x_new = (Q_new[0,1] + Q_new[1,0]) / S
                        y_new = 0.25 * S
                        z_new = (Q_new[1,2] + Q_new[2,1]) / S
                    else:
                        S = np.sqrt(1.0 + Q_new[2,2] - Q_new[0,0] - Q_new[1,1]) * 2
                        w_new = (Q_new[1,0] - Q_new[0,1]) / S
                        x_new = (Q_new[0,2] + Q_new[2,0]) / S
                        y_new = (Q_new[1,2] + Q_new[2,1]) / S
                        z_new = 0.25 * S
                quats_flipped[i] = [w_new, x_new, y_new, z_new]
            aligned_params[:, 6:10] = quats_flipped

    return aligned_params
