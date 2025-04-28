## -*- coding: utf-8 -*-
"""
Generates Figure 2 Recreation from the paper:
"Doppler Analysis of Forward Scattering Radar With Opportunistic Signals From LEO Satellites"
by Defeng Huang (IEEE Access, 2022)
DOI: 10.1109/ACCESS.2022.3214844

NOTE: Plots FULL calculated Doppler shift f_d(t).
Initial object ground position y_G(0) is adjusted to h*cot(theta0) to ensure
collinearity (z_F=0) at t=0, based on constant altitude h.
This setup results in f_d(0)=0 and matches Fig 2.
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

# OBJ_A, OBJ_B not needed for Fig 2
THETA_0_DEG = 65 # Initial elevation angle (degrees)
THETA_0_RAD = np.deg2rad(THETA_0_DEG)

# Speeds for Figure 2
SPEEDS = [300, 150, 0, -150, -300] # Object speeds (m/s, along yG)

# 2. Time vector
t = np.linspace(-1.5, 1.5, 51) # Time vector (secs), fewer points for clarity like Fig 2
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
    # Ensure the argument of sqrt is non-negative
    term_inside_sqrt = np.maximum(term_inside_sqrt, 0)
    numerator = R_E * cos_theta0_sq + np.sqrt(term_inside_sqrt)
    # Ensure the argument of arccos is within [-1, 1]
    alpha_0_arg = np.clip(numerator / r_sat, -1.0, 1.0)
    alpha_0 = np.arccos(alpha_0_arg)
    alpha_t = alpha_0 + d_alpha_dt * t_vec
    cos_alpha_t = np.cos(alpha_t)
    sin_alpha_t = np.sin(alpha_t)
    term1_atan = (r_sat) * sin_alpha_t
    term2_atan = (r_sat) * cos_alpha_t - R_E
    # atan2 handles quadrant ambiguity and term2_atan being zero
    theta_t = np.pi/2 - np.arctan2(term1_atan, term2_atan)

    numerator_dtheta = (r_sat) * R_E * cos_alpha_t - r_sat**2
    denominator_dtheta = r_sat**2 - 2 * r_sat * R_E * cos_alpha_t + R_E**2
    # Avoid division by zero
    denominator_dtheta = np.where(np.abs(denominator_dtheta) < 1e-9, 1e-9, denominator_dtheta)
    # ** CORRECTED LINE 63: Removed trailing backslash **
    dtheta_dt = (numerator_dtheta / denominator_dtheta) * d_alpha_dt

    yS_t_sq = r_sat**2 + R_E**2 - 2 * r_sat * R_E * cos_alpha_t
    # Ensure sqrt argument is non-negative
    yS_t = np.sqrt(np.maximum(yS_t_sq, 0))
    # Avoid division by zero in v_yS_t calculation
    yS_t_safe = np.where(yS_t < 1e-6, 1e-6, yS_t)
    v_yS_numerator = r_sat * R_E * sin_alpha_t * d_alpha_dt
    v_yS_t = v_yS_numerator / yS_t_safe

    return yS_t, v_yS_t, theta_t, dtheta_dt

def calculate_object_kinematics_ground_collinear_t0(t_vec, speed_yg, h_obj, theta0_rad):
    """
    Calculates object kinematics in ground frame (xG, yG, zG).
    Initial yG(0) is set to h_obj * cot(theta0_rad) to ensure z_F(0)=0.
    """
    # Calculate initial offset yG(0) = h * cot(theta0)
    # Handle theta0 = 0 or pi case for cotangent
    if np.abs(np.sin(theta0_rad)) < 1e-9:
        # Avoid division by zero; if theta0 is 0 or pi, offset is infinite/undefined
        # This case shouldn't happen for theta0=65deg
        yG_0 = 0 # Or raise error, but set to 0 for now
    else:
        yG_0 = h_obj / np.tan(theta0_rad) # cot(theta) = 1/tan(theta)

    xG_t = np.zeros_like(t_vec)
    yG_t = yG_0 + speed_yg * t_vec # Apply initial offset
    zG_t = np.full_like(t_vec, h_obj) # Constant altitude

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
    # Avoid division by zero later if r1 is very small
    r1_t = np.where(r1_t < 1e-6, 1e-6, r1_t)

    r2_t = np.sqrt(xF_t**2 + (yF_t - yS_t)**2 + zF_t**2)
    # Avoid division by zero later if r2 is very small
    r2_t = np.where(r2_t < 1e-6, 1e-6, r2_t)
    return r1_t, r2_t

def calculate_fd(lambda_val, r1_t, r2_t, xF_t, yF_t, zF_t, v_xF_t, v_yF_t, v_zF_t, yS_t, v_yS_t):
    """ Calculates FULL Doppler shift of object centroid using Eq. (11). """
    term1_num = xF_t * v_xF_t + yF_t * v_yF_t + zF_t * v_zF_t
    # Use np.divide for safe division
    term1 = np.divide(term1_num, r1_t, out=np.zeros_like(term1_num), where=r1_t!=0)

    term2_num = xF_t * v_xF_t + zF_t * v_zF_t + (yF_t - yS_t) * (v_yF_t - v_yS_t)
    # Use np.divide for safe division
    term2 = np.divide(term2_num, r2_t, out=np.zeros_like(term2_num), where=r2_t!=0)

    fd_t = (1 / lambda_val) * (term1 + term2 - v_yS_t)

    # Handle potential NaN/inf values arising from edge cases or division issues
    fd_t = np.nan_to_num(fd_t, nan=0.0, posinf=0.0, neginf=0.0)
    return fd_t

# --- End Helper Functions ---

# --- Main Calculation Loop ---
plt.figure(figsize=(8, 6))
markers = ['o', '.', '*', 'x', '+'] # Match Fig 2 markers more closely if possible
linestyles = ['-', '--', '-.', ':', '-'] # Match Fig 2 line styles

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

    # Check zF at t=0
    # print(f"Speed: {speed}, zF at t=0: {zF_t[idx_t0]:.4f} m") # Optional debug check

    # Calculate distances and FULL Doppler shift fd
    r1_t, r2_t = calculate_distances(xF_t, yF_t, zF_t, yS_t)
    fd_t_full = calculate_fd(LAMBDA, r1_t, r2_t, xF_t, yF_t, zF_t, v_xF_t, v_yF_t, v_zF_t, yS_t, v_yS_t)

    # --- Plotting FULL Doppler Shift (which should be 0 at t=0) ---
    marker = markers[i % len(markers)]
    ls = linestyles[i % len(linestyles)]
    # Use fewer markers for visual clarity if needed: markevery=5
    # ** CORRECTED LINE 157: Fixed f-string syntax **
    plt.plot(t, fd_t_full, marker=marker, linestyle=ls, label=f'{speed} m/s', markevery=5)

    # Check f_d(0) value
    # print(f"Speed: {speed}, Calculated f_d(0): {fd_t_full[idx_t0]:.4f} Hz") # Optional debug check

# --- Final Plot Formatting ---
plt.xlabel("Time (secs)")
plt.ylabel("Doppler shift in Hz") # Match Fig 2 label
plt.title("Figure 2 Recreation (Initial Pos. Adjusted for t=0 Collinearity)")
plt.legend(title="Speed of flying object")
plt.grid(True)
plt.ylim([-2500, 2500]) # Match Fig 2 Y-axis limit
plt.xlim([-1.5, 1.5]) # Match Fig 2 X-axis limit
plt.tight_layout() # Adjust plot to prevent labels overlapping
plt.show()