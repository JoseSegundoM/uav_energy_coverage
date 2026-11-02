"""acados NMPC OCP for the 18-state quadrotor model.

Direct multiple shooting, HPIPM, Gauss-Newton SQP-RTI; NONLINEAR_LS cost with a
geodesic quaternion attitude error. build_ocp() is used by
paper/scripts/mc_experiments.py. Running this file exports the C code
to c_generated_code/ (as used by src/uav_nmpc_controller).

Run:  python3 generate_acados_c_code.py
"""

import os

# Make acados self-contained regardless of the login shell sourcing .bashrc.
os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))
_acados_lib = os.path.expanduser("~/acados/lib")
os.environ["LD_LIBRARY_PATH"] = (
    _acados_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))

import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from uav_dynamics_casadi import (
    build_model_18states_SX, OMEGA_HOVER, OMEGA_MAX, V_NOM,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Horizon: T=0.2s (N=20) is too short to regulate translational dynamics (a
# quad needs >>0.2s to move 0.5m, so control barely affects position within
# the horizon). N=50 -> T=0.5s converges to the setpoint with <1mm/<1deg error
# while keeping latency P99 < 0.8ms (<< 2ms target). dt=0.01 preserves 100 Hz.
N = 50            # prediction steps
DT = 0.01         # control period [s] -> 100 Hz
T_HORIZON = N * DT  # 0.50 s


# --- quaternion helpers -------------------------------------------------------
def quat_conj(q):
    """Conjugate of quaternion [w,x,y,z]."""
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def quat_mult(a, b):
    """Hamilton product a ⊗ b, convention [w,x,y,z]."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return ca.vertcat(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def build_ocp():
    """Assemble the AcadosOcp for the 18-state NMPC."""
    # --- PASO 1: model (SX, required by acados) — 18-STATE high fidelity ------
    x_sym, u_sym, xdot_expr, params = build_model_18states_SX()
    assert x_sym.shape == (18, 1)
    assert u_sym.shape == (4, 1)
    assert xdot_expr.shape == (18, 1)

    # --- PASO 2: AcadosModel -------------------------------------------------
    model = AcadosModel()
    model.name = "uav_nmpc_18states"
    model.x = x_sym
    model.u = u_sym
    model.f_expl_expr = xdot_expr

    # --- online reference parameters (p), dim 13 -----------------------------
    p_ref = ca.SX.sym("p_ref", 3)    # position reference
    v_ref = ca.SX.sym("v_ref", 3)    # linear velocity reference
    q_ref = ca.SX.sym("q_ref", 4)    # attitude reference quaternion
    w_ref = ca.SX.sym("w_ref", 3)    # body angular-rate reference
    model.p = ca.vertcat(p_ref, v_ref, q_ref, w_ref)

    # --- PASO 4: geodesic attitude error (anti-unwinding) --------------------
    q = x_sym[6:10]
    q_err = quat_mult(quat_conj(q_ref), q)   # q_ref^* ⊗ q
    e_att = q_err[1:4]                        # vector part (dim 3)

    e_pos = x_sym[0:3] - p_ref
    e_vel = x_sym[3:6] - v_ref
    e_omega = x_sym[10:13] - w_ref

    y = ca.vertcat(e_pos, e_vel, e_att, e_omega, u_sym)   # dim 16
    y_e = ca.vertcat(e_pos, e_vel, e_att, e_omega)        # dim 12

    model.cost_y_expr = y
    model.cost_y_expr_e = y_e

    # --- PASO 3: AcadosOcp ---------------------------------------------------
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T_HORIZON
    ocp.parameter_values = np.array([0, 0, 1,        # p_ref
                                     0, 0, 0,        # v_ref
                                     1, 0, 0, 0,     # q_ref (identity)
                                     0, 0, 0],       # w_ref
                                    dtype=float)

    # --- PASO 4 (cont.): cost type -------------------------------------------
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # --- PASO 5: weight matrices --------------------------------------------
    ocp.cost.W = np.diag([10, 10, 10,        # Qp  position
                          0.1, 0.1, 0.1,     # Qdp velocity
                          1, 1, 1,           # Qq  attitude (geodesic)
                          0.5, 0.5, 0.5,     # Qw  angular rate
                          0.01, 0.01, 0.01, 0.01]).astype(float)  # R control
    ocp.cost.W_e = np.diag([10, 10, 10,
                            0.1, 0.1, 0.1,
                            1, 1, 1,
                            0.5, 0.5, 0.5]).astype(float)
    ocp.cost.yref = np.zeros(16)
    ocp.cost.yref_e = np.zeros(12)

    # --- PASO 6: constraints -------------------------------------------------
    # PWM input bounds delta_i in [0, 1]
    ocp.constraints.lbu = np.zeros(4)
    ocp.constraints.ubu = np.ones(4)
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    # state bounds: Omega_i in [0, OMEGA_MAX] (idx 13-16) + V_bat in [14, 16.8]
    ocp.constraints.lbx = np.array([0.0, 0.0, 0.0, 0.0, 14.0])
    ocp.constraints.ubx = np.array(
        [OMEGA_MAX, OMEGA_MAX, OMEGA_MAX, OMEGA_MAX, V_NOM])
    ocp.constraints.idxbx = np.array([13, 14, 15, 16, 17])
    # initial state (18 components, V_bat = 16.8 V, z0 = 1.5 m)
    ocp.constraints.x0 = np.array(
        [0, 0, 1.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         OMEGA_HOVER, OMEGA_HOVER, OMEGA_HOVER, OMEGA_HOVER, V_NOM], dtype=float)

    # --- PASO 7: solver options ----------------------------------------------
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.print_level = 0

    # --- PASO 8: export ------------------------------------------------------
    ocp.code_export_directory = os.path.join(SCRIPT_DIR, "c_generated_code")
    return ocp


if __name__ == "__main__":
    AcadosOcpSolver(build_ocp(), json_file=os.path.join(SCRIPT_DIR, "uav_nmpc_ocp.json"))
    print("C code exported to c_generated_code/")
