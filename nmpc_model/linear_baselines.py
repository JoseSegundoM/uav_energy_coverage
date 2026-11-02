"""Linear baselines: cascade PID and hover LQR.

Both act on the same 18-state nonlinear plant as the NMPC (RK4, dt = 0.01 s)
and output PWM commands delta_i in [0, 1] to the four rotors (X configuration).

Cascade PID: position loop -> desired attitude -> attitude loop -> body torque
-> rotor mixing. LQR: full-state feedback on the 12-state model linearized at
hover, with K from the continuous algebraic Riccati equation
(scipy.linalg.solve_continuous_are).

Used by paper/scripts/mc_experiments.py and baselines_ff.py.
"""
import os
import warnings
warnings.filterwarnings("ignore")


os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))
os.environ["LD_LIBRARY_PATH"] = (
    os.path.expanduser("~/acados/lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))

import numpy as np
import scipy.linalg as sla

from uav_dynamics_casadi import (
    MASS, GRAVITY, ARM_LENGTH, IXX, IYY, IZZ, KF, KM, OMEGA_MAX)

SCRIPT = os.path.dirname(os.path.abspath(__file__))
DT = 0.01
G = GRAVITY
L = ARM_LENGTH
# utilidades cinemáticas
# =============================================================================
def quat_to_euler(q):
    """[w,x,y,z] -> (roll, pitch, yaw) [rad], Tait-Bryan ZYX."""
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def quat_to_R(q):
    """Matriz de rotación body->world de [w,x,y,z]."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def wrench_to_delta(U1, tau):
    """(empuje total U1 [N], par de cuerpo tau=[τx,τy,τz] [N·m]) -> δ_i∈[0,1].

    Invierte la mezcla en X:
        U1 = kf (Σ Ω_i²);  τx = kf l/√2 (-Ω1²+Ω2²+Ω3²-Ω4²)
        τy = kf l/√2 (-Ω1²+Ω2²-Ω3²+Ω4²);  τz = km (Ω1²+Ω2²-Ω3²-Ω4²)
    La matriz de mezcla es ortogonal -> inversa = transpuesta/4.
    """
    a = U1 / KF
    b = tau[0] / (KF * L / np.sqrt(2.0))
    c = tau[1] / (KF * L / np.sqrt(2.0))
    d = tau[2] / KM
    s1 = (a - b - c + d) / 4.0
    s2 = (a + b + c + d) / 4.0
    s3 = (a + b - c - d) / 4.0
    s4 = (a - b + c - d) / 4.0
    s = np.array([s1, s2, s3, s4])
    omega = np.sqrt(np.clip(s, 0.0, None))               # Ω_i [rad/s]
    return np.clip(omega / OMEGA_MAX, 0.0, 1.0)          # δ_i = Ω_i/Ω_max


# =============================================================================
# PID en cascada (posición -> actitud -> par)
# =============================================================================
class CascadePID:
    """PID en cascada para cuadricóptero, sintonizado en hover.

    Lazo externo (posición): error de posición/velocidad -> aceleración deseada
    -> vector de empuje deseado -> actitud deseada (roll/pitch, yaw=0).
    Lazo interno (actitud): error de actitud + amortiguamiento de velocidad
    angular -> par de cuerpo. Mezcla a rotores y saturación a PWM.
    """

    def __init__(self):
        # lazo de posición (P, D, I) por eje
        self.Kp_p = np.array([6.0, 6.0, 8.0])
        self.Kd_p = np.array([4.0, 4.0, 5.0])
        self.Ki_p = np.array([1.2, 1.2, 1.6])
        # lazo de actitud (rad/s² por rad) y amortiguamiento de rate
        self.Kp_att = np.array([55.0, 55.0, 22.0])
        self.Kd_att = np.array([11.0, 11.0, 6.0])
        self.tilt_max = np.deg2rad(35.0)
        self.I = np.array([IXX, IYY, IZZ])
        self.reset()

    def reset(self):
        self._int = np.zeros(3)

    def compute(self, x, p_ref, v_ref):
        p = x[0:3]; v = x[3:6]; q = x[6:10]; omega = x[10:13]
        ep = p_ref - p; ev = v_ref - v
        self._int = np.clip(self._int + ep * DT, -1.5, 1.5)
        a_des = self.Kp_p*ep + self.Kd_p*ev + self.Ki_p*self._int   # acel. deseada
        # vector de empuje deseado en mundo (compensa gravedad)
        F_des = MASS * (a_des + np.array([0.0, 0.0, G]))
        Fn = np.linalg.norm(F_des) + 1e-9
        zb_des = F_des / Fn
        # actitud deseada con yaw=0 a partir del eje z deseado
        roll_des = np.arcsin(np.clip(-zb_des[1], -1, 1))
        pitch_des = np.arctan2(zb_des[0], zb_des[2])
        roll_des = np.clip(roll_des, -self.tilt_max, self.tilt_max)
        pitch_des = np.clip(pitch_des, -self.tilt_max, self.tilt_max)
        # empuje sobre el eje z del cuerpo actual
        R = quat_to_R(q)
        U1 = float(F_des @ R[:, 2])
        U1 = max(U1, 0.2 * MASS * G)
        # lazo interno de actitud
        roll, pitch, yaw = quat_to_euler(q)
        e_att = np.array([roll_des - roll, pitch_des - pitch, 0.0 - yaw])
        ang_acc = self.Kp_att*e_att - self.Kd_att*omega
        tau = self.I * ang_acc                              # par de cuerpo [N·m]
        return wrench_to_delta(U1, tau)


# =============================================================================
# LQR de realimentación de estado (modelo linealizado en hover)
# =============================================================================
class HoverLQR:
    """LQR sobre el modelo lineal de 12 estados [p, v, euler, ω] en hover.

    Dinámica linealizada (yaw=0, ángulos pequeños, U1≈mg):
        ṗ = v ;  v̇x = g·θ ;  v̇y = -g·φ ;  v̇z = ΔU1/m
        φ̇ = p ;  θ̇ = q ;  ψ̇ = r
        ṗ = τx/Ixx ; q̇ = τy/Iyy ; ṙ = τz/Izz
    Entradas u = [ΔU1, τx, τy, τz]. K se obtiene de la ARE continua.
    """

    def __init__(self):
        A = np.zeros((12, 12)); B = np.zeros((12, 4))
        # ṗ = v
        A[0, 3] = A[1, 4] = A[2, 5] = 1.0
        # v̇x = g θ (idx 7), v̇y = -g φ (idx 6)
        A[3, 7] = G; A[4, 6] = -G
        # euleṙ = ω
        A[6, 9] = A[7, 10] = A[8, 11] = 1.0
        # v̇z = ΔU1/m
        B[5, 0] = 1.0 / MASS
        # ω̇ = τ/I
        B[9, 1] = 1.0 / IXX; B[10, 2] = 1.0 / IYY; B[11, 3] = 1.0 / IZZ
        # pesos: penaliza posición y actitud; entradas moderadas
        Q = np.diag([18, 18, 22, 2, 2, 3, 6, 6, 4, 0.3, 0.3, 0.3]).astype(float)
        Rm = np.diag([0.7, 5.0, 5.0, 5.0]).astype(float)
        P = sla.solve_continuous_are(A, B, Q, Rm)
        self.K = np.linalg.solve(Rm, B.T @ P)              # u = -K (x - x_ref)
        self.A, self.B = A, B

    def reset(self):
        pass

    def compute(self, x, p_ref, v_ref):
        p = x[0:3]; v = x[3:6]; q = x[6:10]; omega = x[10:13]
        roll, pitch, yaw = quat_to_euler(q)
        st = np.array([p[0], p[1], p[2], v[0], v[1], v[2],
                       roll, pitch, yaw, omega[0], omega[1], omega[2]])
        ref = np.array([p_ref[0], p_ref[1], p_ref[2],
                        v_ref[0], v_ref[1], v_ref[2], 0, 0, 0, 0, 0, 0])
        u = -self.K @ (st - ref)                            # [ΔU1, τx, τy, τz]
        U1 = MASS * G + u[0]
        U1 = max(U1, 0.2 * MASS * G)
        return wrench_to_delta(U1, u[1:4])


def make_baselines():
    """Devuelve {'pid': CascadePID, 'lqr': HoverLQR} ya instanciados."""
    return {"pid": CascadePID(), "lqr": HoverLQR()}


