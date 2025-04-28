# -*- coding: utf-8 -*-
"""
Generates Figure 4 Recreation from the paper:
"Doppler Analysis of Forward Scattering Radar With Opportunistic Signals From LEO Satellites"
by Defeng Huang (IEEE Access, 2022)
DOI: 10.1109/ACCESS.2022.3214844

NOTE: Plots Absolute Value of FULL calculated Doppler shift |f_d(t)|
and the original, non-offset bounds bound1(t) and bound2(t).
Initial object ground position y_G(0) is adjusted to h*cot(theta0) to ensure
collinearity (z_F=0) at t=0. This setup results in f_d(0)=0.
Legend labels match Fig. 4.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Constants & Parameters
R_E = 6.371e6 # Earth radius (m)
G = 6.6743e-11 # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.9722e24 # Earth mass (kg)
LAMBDA = 0.01 # Wavelength (m) -> fc = 30 GHz
K = 2 * np.pi / LAMBDA # Wave number (rad/m)
H_SAT = 550e3 # Satellite altitude (m)
H_OBJ = 10e3 # Object altitude (m)
OBJ_A = 20 # Object width (m, in x direction)
OBJ_B = 40 # Object length (m, in yG direction)
THETA_0_DEG = 65 # Initial elevation angle (degrees)
THETA_0_RAD = np.deg2rad(THETA_0_DEG)
# Speeds for Figure 4
SPEEDS = [300, -150, -300]

# 2. Time vector
t = np.linspace(-1.5, 1.5, 301) # Time vector (secs)
idx_t0 = np.argmin(np.abs(t)) # Index for t=0

# --- Helper Functions ---
def calculate_satellite_kinematics(t_vec, h_sat, theta_0_rad):
    """ Calculates satellite position, velocity, and angles over time. """
    r_sat = h_sat + R_E
    T_Sat = 2 * np.pi * np.sqrt(r_sat**3 / (G * M_EARTH))
    d_alpha_dt = 2 * np.pi / T_Sat
    cos_theta0_sq = np.cos(theta_0_rad)**2
    sin_theta0_sq = np.sin(theta_0_rad)**2
    term_inside_sqrt = sin_theta0_sq * (r_sat**2 - R_E**2 * cos_theta0_sq)
    term_inside_sqrt = np.maximum(term_inside_sqrt, 0)
    numerator = R_E * cos_theta0_sq + np.sqrt(term_inside_sqrt)
    alpha_0_arg = np.clip(numerator / r_sat, -1.0, 1.0)
    alpha_0 = np.arccos(alpha_0_arg)
    alpha_t = alpha_0 + d_alpha_dt * t_vec
    cos_alpha_t = np.cos(alpha_t)
    sin_alpha_t = np.sin(alpha_t)
    term1_atan = (r_sat) * sin_alpha_t
    term2_atan = (r_sat) * cos_alpha_t - R_E
    theta_t = np.pi/2 - np.arctan2(term1_atan, term2_atan)
    numerator_dtheta = (r_sat) * R_E * cos_alpha_t - r_sat**2
    denominator_dtheta = r_sat**2 - 2 * r_sat * R_E * cos_alpha_t + R_E**2
    denominator_dtheta = np.where(np.abs(denominator_dtheta) < 1e-9, 1e-9, denominator_dtheta)
    dtheta_dt = (numerator_dtheta / denominator_dtheta) * d_alpha_dt
    yS_t_sq = r_sat**2 + R_E**2 - 2 * r_sat * R_E * cos_alpha_t
    yS_t = np.sqrt(np.maximum(yS_t_sq, 0))
    yS_t_safe = np.where(yS_t < 1e-6, 1e-6, yS_t)
    v_yS_numerator = r_sat * R_E * sin_alpha_t * d_alpha_dt
    v_yS_t = v_yS_numerator / yS_t_safe
    return yS_t, v_yS_t, theta_t, dtheta_dt

def calculate_object_kinematics_ground_collinear_t0(t_vec, speed_yg, h_obj, theta0_rad):
    """
    Calculates object kinematics in ground frame (xG, yG, zG).
    Initial yG(0) is set to h_obj * cot(theta0_rad) to ensure z_F(0)=0.
    """
    if np.abs(np.sin(theta0_rad)) < 1e-9:
        yG_0 = 0
    else:
        yG_0 = h_obj / np.tan(theta0_rad) # h * cot(theta0)

    xG_t = np.zeros_like(t_vec)
    yG_t = yG_0 + speed_yg * t_vec
    zG_t = np.full_like(t_vec, h_obj)
    v_xG_t = np.zeros_like(t_vec)
    v_yG_t = np.full_like(t_vec, speed_yg)
    v_zG_t = np.zeros_like(t_vec)
    return xG_t, yG_t, zG_t, v_xG_t, v_yG_t, v_zG_t

def transform_to_rotational_frame(xG_t, yG_t, zG_t, v_xG_t, v_yG_t, v_zG_t, theta_t, dtheta_dt):
    """ Transforms object kinematics from ground frame to rotational frame. """
    xF_t = xG_t
    cos_theta = np.cos(theta_t)
    sin_theta = np.sin(theta_t)
    yF_t = cos_theta * yG_t + sin_theta * zG_t
    zF_t = -sin_theta * yG_t + cos_theta * zG_t
    v_xF_t = v_xG_t
    term1_vy = cos_theta * v_yG_t + sin_theta * v_zG_t
    term2_vy = (-sin_theta * yG_t + cos_theta * zG_t) * dtheta_dt
    v_yF_t = term1_vy + term2_vy
    term1_vz = -sin_theta * v_yG_t + cos_theta * v_zG_t
    term2_vz = (-cos_theta * yG_t - sin_theta * zG_t) * dtheta_dt
    v_zF_t = term1_vz + term2_vz
    return xF_t, yF_t, zF_t, v_xF_t, v_yF_t, v_zF_t

def calculate_distances(xF_t, yF_t, zF_t, yS_t):
    """ Calculates distances r1 (object-RX) and r2 (object-TX)."""
    r1_t = np.sqrt(xF_t**2 + yF_t**2 + zF_t**2)
    r1_t = np.where(r1_t < 1e-6, 1e-6, r1_t)
    r2_t = np.sqrt(xF_t**2 + (yF_t - yS_t)**2 + zF_t**2)
    r2_t = np.where(r2_t < 1e-6, 1e-6, r2_t)
    return r1_t, r2_t

def calculate_fd(lambda_val, r1_t, r2_t, xF_t, yF_t, zF_t, v_xF_t, v_yF_t, v_zF_t, yS_t, v_yS_t):
    """ Calculates FULL Doppler shift of object centroid using Eq. (11). """
    term1_num = xF_t * v_xF_t + yF_t * v_yF_t + zF_t * v_zF_t
    term1 = np.divide(term1_num, r1_t, out=np.zeros_like(term1_num), where=r1_t!=0)
    term2_num = xF_t * v_xF_t + zF_t * v_zF_t + (yF_t - yS_t) * (v_yF_t - v_yS_t)
    term2 = np.divide(term2_num, r2_t, out=np.zeros_like(term2_num), where=r2_t!=0)
    fd_t = (1 / lambda_val) * (term1 + term2 - v_yS_t)
    fd_t = np.nan_to_num(fd_t, nan=0.0, posinf=0.0, neginf=0.0)
    return fd_t

def calculate_eta(k_val, r1_t, r2_t):
   """ Calculates eta(t) = k * (1/r1 + 1/r2)."""
   eta_t = k_val * (1 / r1_t + 1 / r2_t)
   eta_t = np.nan_to_num(eta_t, nan=0.0, posinf=0.0, neginf=0.0)
   return eta_t

def calculate_fxz(fd_t, eta_t, v_xF_t, v_zF_t, x_prime, z_prime):
  """ Calculates Doppler shift for point (x', z') using Eq. (14) approx."""
  doppler_point = fd_t + (eta_t / (2 * np.pi)) * (x_prime * v_xF_t + z_prime * v_zF_t)
  return doppler_point

def get_object_projection_corners(a, b, theta_t):
    """ Calculates the corners (x', z') of the object's projection. """
    num_times = len(theta_t)
    half_a = a / 2.0
    half_z_proj = (b / 2.0) * np.sin(theta_t)
    corners_time = np.zeros((num_times, 4, 2))
    corners_time[:, 0, 0] = -half_a; corners_time[:, 0, 1] = -half_z_proj
    corners_time[:, 1, 0] = +half_a; corners_time[:, 1, 1] = -half_z_proj
    corners_time[:, 2, 0] = -half_a; corners_time[:, 2, 1] = +half_z_proj
    corners_time[:, 3, 0] = +half_a; corners_time[:, 3, 1] = +half_z_proj
    return corners_time
# --- End Helper Functions ---


# --- Main Calculation Loop ---
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r'] # Colors for speeds [300, -150, -300]
lines = []
min_val = float('inf')
max_val = float('-inf')

for i, speed in enumerate(SPEEDS):
    # Calculate satellite kinematics
    yS_t, v_yS_t, theta_t, dtheta_dt = calculate_satellite_kinematics(t, H_SAT, THETA_0_RAD)
    # Calculate object kinematics with ADJUSTED initial yG(0) for collinearity
    xG_t, yG_t, zG_t, v_xG_t, v_yG_t, v_zG_t = calculate_object_kinematics_ground_collinear_t0(
        t, speed, H_OBJ, THETA_0_RAD
    )
    # Transform to rotational frame using STANDARD transform
    xF_t, yF_t, zF_t, v_xF_t, v_yF_t, v_zF_t = transform_to_rotational_frame(
        xG_t, yG_t, zG_t, v_xG_t, v_yG_t, v_zG_t, theta_t, dtheta_dt
    )

    # Calculate distances and FULL Doppler shift fd
    r1_t, r2_t = calculate_distances(xF_t, yF_t, zF_t, yS_t)
    fd_t_full = calculate_fd(LAMBDA, r1_t, r2_t, xF_t, yF_t, zF_t, v_xF_t, v_yF_t, v_zF_t, yS_t, v_yS_t)

    # Calculate eta
    eta_t = calculate_eta(K, r1_t, r2_t)

    # Calculate bounds based on FULL fd_t
    corners_t = get_object_projection_corners(OBJ_A, OBJ_B, theta_t)
    bound1_t_full = np.zeros_like(t)
    bound2_t_full = np.zeros_like(t)

    skip_bounds = np.any(np.isnan(xF_t)) or np.any(np.isnan(v_zF_t)) or \
                  np.any(np.isinf(xF_t)) or np.any(np.isinf(v_zF_t)) or \
                  np.any(np.isnan(fd_t_full)) or np.any(np.isinf(fd_t_full)) or \
                  np.any(np.isnan(eta_t)) or np.any(np.isinf(eta_t))

    if not skip_bounds:
        for j in range(len(t)):
            current_fd = fd_t_full[j]; current_eta = eta_t[j]
            current_vx = v_xF_t[j]; current_vz = v_zF_t[j]
            if np.isnan(current_fd) or np.isinf(current_fd) or \
               np.isnan(current_eta) or np.isinf(current_eta) or \
               np.isnan(current_vz) or np.isinf(current_vz):
                bound1_t_full[j] = np.nan; bound2_t_full[j] = np.nan; continue
            f_corners = []
            for k in range(4):
                x_prime = corners_t[j, k, 0]; z_prime = corners_t[j, k, 1]
                f_corner = calculate_fxz(current_fd, current_eta, current_vx, current_vz, x_prime, z_prime)
                f_corners.append(f_corner)
            f_corners = np.array(f_corners)
            if np.any(np.isnan(f_corners)) or np.any(np.isinf(f_corners)):
                 bound1_t_full[j] = np.nan; bound2_t_full[j] = np.nan
            else:
                 abs_f_corners = np.abs(f_corners) # Use absolute value for bounds calculation
                 bound1_t_full[j] = np.max(abs_f_corners)
                 bound2_t_full[j] = np.min(abs_f_corners)
    else:
         bound1_t_full[:] = np.nan; bound2_t_full[:] = np.nan


    # --- Plotting |fd_full| and FULL Bounds ---
    color = colors[i % len(colors)]
    # Plot absolute value of full fd
    fd_t_full_abs = np.abs(fd_t_full) # <--- APPLY ABSOLUTE VALUE HERE
    line_fd, = plt.plot(t, fd_t_full_abs, color=color, linestyle='-', label='Doppler shift') # Use simple label
    # Plot full bounds (already non-negative)
    line_b1, = plt.plot(t, bound1_t_full, color=color, linestyle=':', label='bound 1') # Use simple label
    line_b2, = plt.plot(t, bound2_t_full, color=color, linestyle='--', label='bound 2') # Use simple label

    # Update min/max for y-limit calculation (use absolute fd)
    current_min = np.nanmin([np.nanmin(a) if np.any(np.isfinite(a)) else float('inf') for a in [fd_t_full_abs, bound1_t_full, bound2_t_full]])
    current_max = np.nanmax([np.nanmax(a) if np.any(np.isfinite(a)) else float('-inf') for a in [fd_t_full_abs, bound1_t_full, bound2_t_full]])
    min_val = np.nanmin([min_val, max(0, current_min)]) # Ensure min >= 0
    max_val = np.nanmax([max_val, current_max])


    # Store one set of lines for the legend
    if i == 0:
        lines.extend([line_fd, line_b1, line_b2])
    # Add text annotation
    time_idx_annot = np.argmin(np.abs(t - (-1.0)))
    valid_y_annot = bound1_t_full[np.isfinite(bound1_t_full)]
    if len(valid_y_annot) > 0:
         y_annot_base = bound1_t_full[time_idx_annot]
         plot_ymin, plot_ymax = -100, 2500 # Use intended plot range
         if np.isfinite(y_annot_base):
             # Adjust annotation based on where the base value lies
             if y_annot_base < (plot_ymax + plot_ymin) / 2 : # Check if point is in lower half
                  annot_y_pos = y_annot_base + (plot_ymax - plot_ymin) * 0.05 # Add offset if low
             else:
                  annot_y_pos = y_annot_base - (plot_ymax - plot_ymin) * 0.1 # Subtract offset if high
         else: # If value at index is NaN, find nearest valid point
             valid_t_indices = np.where(np.isfinite(bound1_t_full))[0]
             if len(valid_t_indices)>0:
                 closest_valid_idx = valid_t_indices[np.argmin(np.abs(t[valid_t_indices] - t[time_idx_annot]))]
                 annot_y_pos = bound1_t_full[closest_valid_idx] + (plot_ymax - plot_ymin) * 0.05
             else: # Fallback if no valid y points exist
                 annot_y_pos = 100
    else: # Fallback if no valid y points exist
         annot_y_pos = 100
    annot_y_pos = np.clip(annot_y_pos, 50, 2450) # Keep annotation within plot bounds

    plt.text(t[time_idx_annot], annot_y_pos , f'{speed} m/s', color=color, ha='center')


# --- Final Plot Formatting ---
plt.xlabel("Time (secs)")
plt.ylabel("Doppler shift (Hz)") # Match Fig 4 label
plt.title("Figure 4 Recreation (Collinear t=0, Plotting $|f_d(t)|$)")
# Create a simplified legend matching Fig 4
legend_labels = ['Doppler shift', 'bound 1', 'bound 2']
plt.legend(handles=lines, labels=legend_labels, loc='upper right')
plt.grid(True)
# Set Y limits matching Figure 4
plt.ylim([-100, 2500])
plt.xlim([-1.5, 1.5])
plt.tight_layout()
plt.show()