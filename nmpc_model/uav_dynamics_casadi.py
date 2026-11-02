"""Quadrotor dynamics in CasADi (18 states).

build_model_18states() is the plant simulated in the Monte-Carlo campaigns of
HRFEST 2026 paper #235, and also the model inside the NMPC OCP
(generate_acados_c_code.py). States (18):
    [x, y, z, dx, dy, dz, q0, q1, q2, q3, p, q, r,
     Omega1, Omega2, Omega3, Omega4, V_bat]
Includes quaternion kinematics, translational drag (CD), rotational drag
(CDw), gyroscopic and blade-flapping torques, ground effect kf_eff(z) and
battery voltage dynamics.

Quaternion convention: [q0, q1, q2, q3] = [w, x, y, z] everywhere.
"""

import casadi as ca

# --- Physical parameters (single source of truth: docs/TECHNICAL_SPEC.md) ----
# Mass and geometry
MASS = 0.80              # kg
GRAVITY = 9.81           # m/s^2
ARM_LENGTH = 0.175       # m (CoM-to-rotor diagonal arm)

# Inertias
IXX = 4.9e-3             # kg*m^2 (roll)
IYY = 4.9e-3             # kg*m^2 (pitch)
IZZ = 8.8e-3             # kg*m^2 (yaw)
JR = 6.0e-5              # kg*m^2 (rotor axial inertia)

# Propulsion coefficients
KF = 2.3e-5              # N*s^2/rad^2 (thrust)
KM = 7.0e-7              # N*m*s^2/rad^2 (reactive torque)

# Actuators
TAU_M = 0.03             # s (electromechanical time constant)
OMEGA_MAX = 1500.0       # rad/s

# Battery
V_NOM = 16.8             # V (4S LiPo nominal)
C_BAT = 3.0              # Ah (capacity)
ETA = 0.85              # - (propulsion efficiency)
ALPHA = OMEGA_MAX / V_NOM  # rad/(s*V)

# Aerodynamics
CD = 0.15                # kg/m (translational drag)
CD_W = 5.0e-4            # kg*m^2/rad (rotational drag)
H_GE = 2 * ARM_LENGTH    # m (ground-effect characteristic height)
A_BF = 1.0e-6            # N*s/rad (blade flapping)

# Control constraints
DELTA_MIN = 0.0
DELTA_MAX = 1.0
OMEGA_MIN = 0.0

# Hover operating point (derived from physical parameters above)
#   Omega_hover = sqrt(m*g / (4*kf)) -> rotor speed that cancels weight
#   delta_hover = Omega_hover / Omega_max -> normalized PWM at hover
OMEGA_HOVER = 292.0              # rad/s  (sqrt(0.8*9.81/(4*2.3e-5)) ~ 292.0)
DELTA_HOVER = 0.1947            # PWM    (292.0 / 1500 ~ 0.1947)


def _params_dict():
    """Collect all numeric parameters into a dictionary."""
    return {
        "MASS": MASS, "GRAVITY": GRAVITY, "ARM_LENGTH": ARM_LENGTH,
        "IXX": IXX, "IYY": IYY, "IZZ": IZZ, "JR": JR,
        "KF": KF, "KM": KM,
        "TAU_M": TAU_M, "OMEGA_MAX": OMEGA_MAX,
        "V_NOM": V_NOM, "C_BAT": C_BAT, "ETA": ETA, "ALPHA": ALPHA,
        "CD": CD, "CD_W": CD_W, "H_GE": H_GE, "A_BF": A_BF,
        "DELTA_MIN": DELTA_MIN, "DELTA_MAX": DELTA_MAX, "OMEGA_MIN": OMEGA_MIN,
    }


def build_model_18states(sym=ca.MX):
    """Build the 18-state high-fidelity model (plant AND, now, the NMPC model).

    Args:
        sym: CasADi symbolic backend (ca.MX or ca.SX). acados requires SX.

    Returns:
        x (sym):      state vector  (18,)
        u (sym):      control vector (4,) — normalized PWM delta_i in [0,1]
        xdot (sym):   symbolic dynamics f(x, u) (18,)
        params (dict):  physical parameters used by the model
    """
    # --- symbolic state and control ------------------------------------------
    x = sym.sym("x18", 18)
    u = sym.sym("u", 4)

    # unpack state (quaternion convention [q0,q1,q2,q3] = [w,x,y,z])
    px, py, pz = x[0], x[1], x[2]
    dx, dy, dz = x[3], x[4], x[5]
    q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
    p, q_ang, r = x[10], x[11], x[12]
    Om1, Om2, Om3, Om4 = x[13], x[14], x[15], x[16]
    V_bat = x[17]

    d1, d2, d3, d4 = u[0], u[1], u[2], u[3]

    # --- ground effect: kf_eff(z) --------------------------------------------
    # Physical ground effect: thrust INCREASES near the floor.
    #   kf_eff = kf * (1 + (h_ge/z)^2)  -> kf_eff > kf as z -> h_ge
    # Guard against z -> 0 with if_else AND a clamped z inside the formula:
    # acados auto-diffs BOTH if_else branches, so the true branch must stay
    # finite even at the (zero) initial guess, or the QP gets NaN.
    z_safe = ca.fmax(pz, 0.1)
    kf_eff_formula = KF * (1.0 + (H_GE / z_safe) ** 2)
    kf_eff = ca.if_else(pz > 0.1, kf_eff_formula, KF)

    # --- virtual inputs and mixing (X configuration) -------------------------
    Om1_2, Om2_2, Om3_2, Om4_2 = Om1**2, Om2**2, Om3**2, Om4**2
    U1 = kf_eff * (Om1_2 + Om2_2 + Om3_2 + Om4_2)
    U2 = (kf_eff * ARM_LENGTH / ca.sqrt(2)) * (-Om1_2 + Om2_2 + Om3_2 - Om4_2)
    U3 = (kf_eff * ARM_LENGTH / ca.sqrt(2)) * (-Om1_2 + Om2_2 - Om3_2 + Om4_2)
    U4 = KM * (Om1_2 + Om2_2 - Om3_2 - Om4_2)

    Omega_res = Om1 + Om2 - Om3 - Om4
    tau_bf_x = A_BF * Omega_res * q_ang
    tau_bf_y = -A_BF * Omega_res * p

    # --- 1. translational kinematics -----------------------------------------
    xdot_pos = ca.vertcat(dx, dy, dz)

    # --- 2. translational dynamics (quaternion + translational drag CD) ------
    # +1e-8 regularizes ||v||: the bare sqrt has an undefined Jacobian
    # (v/||v|| = 0/0) at zero velocity, which NaNs the acados sensitivities at
    # the zero initial guess. Negligible (sqrt floor ~1e-4 m/s) physically.
    norm_v = ca.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
    ddx = (U1 / MASS) * 2 * (q1 * q3 + q0 * q2) - (CD / MASS) * norm_v * dx
    ddy = (U1 / MASS) * 2 * (q2 * q3 - q0 * q1) - (CD / MASS) * norm_v * dy
    ddz = (U1 / MASS) * (1 - 2 * (q1**2 + q2**2)) - GRAVITY - (CD / MASS) * norm_v * dz
    xdot_vel = ca.vertcat(ddx, ddy, ddz)

    # --- 3. quaternion kinematics --------------------------------------------
    dq0 = 0.5 * (-q1 * p - q2 * q_ang - q3 * r)
    dq1 = 0.5 * (q0 * p - q3 * q_ang + q2 * r)
    dq2 = 0.5 * (q3 * p + q0 * q_ang - q1 * r)
    dq3 = 0.5 * (-q2 * p + q1 * q_ang + q0 * r)
    xdot_quat = ca.vertcat(dq0, dq1, dq2, dq3)

    # --- 4. rotational dynamics (gyroscopic + blade flapping + rot. drag) ----
    dp = ((IYY - IZZ) / IXX) * q_ang * r - (JR / IXX) * Omega_res * q_ang \
        + (U2 + tau_bf_x) / IXX - (CD_W / IXX) * p
    dq = ((IZZ - IXX) / IYY) * p * r + (JR / IYY) * Omega_res * p \
        + (U3 + tau_bf_y) / IYY - (CD_W / IYY) * q_ang
    dr = ((IXX - IYY) / IZZ) * p * q_ang + U4 / IZZ - (CD_W / IZZ) * r
    xdot_omega = ca.vertcat(dp, dq, dr)

    # --- 5. actuator dynamics (battery dependent) ----------------------------
    dOm1 = (1.0 / TAU_M) * (ALPHA * V_bat * d1 - Om1)
    dOm2 = (1.0 / TAU_M) * (ALPHA * V_bat * d2 - Om2)
    dOm3 = (1.0 / TAU_M) * (ALPHA * V_bat * d3 - Om3)
    dOm4 = (1.0 / TAU_M) * (ALPHA * V_bat * d4 - Om4)
    xdot_rotor = ca.vertcat(dOm1, dOm2, dOm3, dOm4)

    # --- 6. battery dynamics -------------------------------------------------
    # V_safe guards the 1/V_bat term against the (zero) initial guess acados
    # uses; identical to V_bat for any physical state (V_bat >= 14 V).
    V_safe = ca.fmax(V_bat, 1.0)
    dV_bat = -(V_NOM * KM) / (3600.0 * C_BAT * ETA * V_safe) \
        * (Om1**3 + Om2**3 + Om3**3 + Om4**3)

    xdot = ca.vertcat(
        xdot_pos, xdot_vel, xdot_quat, xdot_omega, xdot_rotor, dV_bat
    )

    return x, u, xdot, _params_dict()


def build_model_18states_SX():
    """18-state high-fidelity model built with CasADi SX (required by acados)."""
    return build_model_18states(sym=ca.SX)


# --- RK4 integrator -----------------------------------------------------------
def rk4_step(f, x, u, dt):
    """One explicit RK4 step of xdot = f(x, u) with fixed dt (matches NMPC dt)."""
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


