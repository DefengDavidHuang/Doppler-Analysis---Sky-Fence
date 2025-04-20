# Fig. 2 in the paper 

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G   = 6.6743e-11         # Universal gravitational constant (m^3 kg^−1 s^−2)
M_E = 5.9722e24          # Earth mass (kg)
r_E = 6.371e6            # Earth radius (m)

# Satellite parameters
h_sat = 550e3                     # Satellite altitude (m)
r_s   = r_E + h_sat               # Orbital radius (m)
omega = np.sqrt(G * M_E / r_s**3) # Orbital angular speed (rad/s)

# Simulation setup
h_target = 10e3                   # Object height above ground (m)
theta0   = np.deg2rad(65)         # Initial satellite elevation (rad)
lambda_c = 0.01                   # Wavelength (m)

# Compute initial geometry
yG0    = h_target / np.tan(theta0)
alpha0 = np.arccos((r_E * np.cos(theta0)) / r_s) - theta0

# Time vector and object speeds
t      = np.linspace(-1.5, 1.5, 300)
speeds = [300, 150, 0, -150, -300]

# Precompute satellite angles & rates
alpha   = alpha0 + omega * t
theta_t = np.pi/2 - np.arctan2(r_s * np.sin(alpha),
                               r_s * np.cos(alpha) - r_E)
dtheta_dt = np.gradient(theta_t, t)
yS      = np.sqrt(r_s**2 + r_E**2 - 2*r_E*r_s*np.cos(alpha))
vyS     = - omega * r_s * np.cos(theta_t + alpha)

# Doppler results container
doppler = {}

for v in speeds:
    # Ground‐frame object motion
    yG  = yG0 + v * t
    zG  = np.full_like(t, h_target)
    vyG = np.full_like(t, v)
    vzG = np.zeros_like(t)

    # Rotate into satellite‐centric frame
    y_rot =  np.cos(theta_t)*yG + np.sin(theta_t)*zG
    z_rot = -np.sin(theta_t)*yG + np.cos(theta_t)*zG
    vy_rot = (np.cos(theta_t)*vyG + np.sin(theta_t)*vzG
              + (-np.sin(theta_t)*yG + np.cos(theta_t)*zG)*dtheta_dt)
    vz_rot = (-np.sin(theta_t)*vyG + np.cos(theta_t)*vzG
              + (-np.cos(theta_t)*yG - np.sin(theta_t)*zG)*dtheta_dt)

    # Path lengths
    r1 = np.hypot(y_rot,     z_rot)
    r2 = np.hypot(y_rot - yS, z_rot)

    # Range‐rate (Eq. 11)
    dr1 = (y_rot*vy_rot + z_rot*vz_rot) / r1
    dr2 = ((y_rot - yS)*(vy_rot - vyS) + z_rot*vz_rot) / r2

    # Doppler shift
    doppler[v] = (dr1 + dr2 - vyS) / lambda_c

# Plot
plt.figure(figsize=(8,5))
for v in speeds:
    plt.plot(t, doppler[v], label=f'{v} m/s')
plt.xlabel('Time (s)')
plt.ylabel('Doppler shift $f_d(t)$ (Hz)')
plt.title('Doppler Shifts via Eq. (11)\nTime Range: –1.5 to 1.5 s')
plt.legend(title='Speed')
plt.grid(True)
plt.tight_layout()
plt.show()