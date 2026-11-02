#!/usr/bin/env python3
"""power_model.py — Potencia del multirrotor en vuelo de avance.

Sustituye al proxy E = P_hover * t. Un revisor tiene razón al objetar que un
barrido agrícola pasa casi todo el tiempo en vuelo de avance, donde la
aerodinámica no es la del hover, así que aquí se implementa el modelo de
teoría de cantidad de movimiento con los tres términos:

    P_mech(V) = P_ind(V) + P_perfil(V) + P_parasita(V)

  - Inducida: v_i resuelve  v_i = v_h^2 / sqrt(V^2 + v_i^2)   (vuelo de canto,
    ángulo de ataque pequeño). En avance la estela se "aplana" y v_i BAJA, así
    que la potencia inducida cae con la velocidad.
  - Perfil: P_p0 * (1 + 4.65 mu^2), con mu = V / (Omega R) la relación de avance.
  - Parásita: F_drag * V = CD * V^3, usando EL MISMO CD de arrastre traslacional
    que emplea la planta del MIL. No se introduce un coeficiente nuevo.

CALIBRACIÓN. El modelo de la planta define el par reactivo por rotor
(Q = KM*w^2), luego la potencia de eje en hover es P_h = 4*KM*w_h^3, un dato
duro. Pero NO define el radio del rotor, y la teoría de cantidad de movimiento
lo necesita. En vez de inventar un radio, se fija el área de disco efectiva
para que el modelo REPRODUZCA esa potencia de hover, repartiendo P_h entre
inducida y perfil con una fracción de perfil típica de rotores pequeños. Así el
modelo es consistente con la planta en V=0 por construcción, y la única
suposición libre queda declarada.

Limitación que permanece: no modela el coste energético del viento (el
controlador gasta empuje extra rechazándolo). Para eso hace falta integrar la
potencia a partir de las revoluciones REALES de la simulación, que es lo que
hace `energy_from_rollout()`.
"""
from __future__ import annotations

import numpy as np

# --- parámetros de la planta (nmpc_model/uav_dynamics_casadi.py) -------
MASS = 0.80           # kg
GRAVITY = 9.81        # m/s^2
KM = 7.0e-7           # N*m*s^2/rad^2  par reactivo
KF = 2.3e-5           # N*s^2/rad^2    empuje
OMEGA_HOVER = 292.0   # rad/s
ETA = 0.85            # rendimiento de propulsión (eléctrico -> eje)
CD = 0.15             # kg/m, arrastre traslacional de la planta
RHO = 1.225           # kg/m^3

K_IND = 1.15          # factor de corrección de potencia inducida (estándar)
PROFILE_FRAC = 0.25   # fracción de la potencia de hover atribuida al perfil

# --- calibración a partir de la potencia de hover de la planta ---------------
P_SHAFT_HOVER = 4.0 * KM * OMEGA_HOVER ** 3          # W de eje
P_ELEC_HOVER = P_SHAFT_HOVER / ETA                   # W eléctricos
THRUST_HOVER = MASS * GRAVITY                        # N

P_PROFILE_0 = PROFILE_FRAC * P_SHAFT_HOVER
_P_IND_HOVER = P_SHAFT_HOVER - P_PROFILE_0
V_H = _P_IND_HOVER / (K_IND * THRUST_HOVER)          # m/s, velocidad inducida
DISC_AREA = THRUST_HOVER / (2.0 * RHO * V_H ** 2)    # m^2, área efectiva total
ROTOR_R = float(np.sqrt(DISC_AREA / (4.0 * np.pi)))  # m, radio equivalente


def induced_velocity(V: np.ndarray | float) -> np.ndarray:
    """Resuelve v_i = v_h^2 / sqrt(V^2 + v_i^2) por punto fijo."""
    V = np.atleast_1d(np.asarray(V, float))
    vi = np.full_like(V, V_H)
    for _ in range(60):                      # converge en ~10 iteraciones
        vi = V_H ** 2 / np.sqrt(V ** 2 + vi ** 2)
    return vi


def power_electrical(V: np.ndarray | float, thrust: np.ndarray | float | None = None
                     ) -> np.ndarray:
    """Potencia eléctrica [W] en vuelo de avance nivelado a velocidad V."""
    V = np.atleast_1d(np.asarray(V, float))
    T = THRUST_HOVER if thrust is None else np.asarray(thrust, float)
    p_ind = K_IND * T * induced_velocity(V)
    mu = V / (OMEGA_HOVER * ROTOR_R)
    p_prof = P_PROFILE_0 * (1.0 + 4.65 * mu ** 2)
    p_par = CD * V ** 3
    return (p_ind + p_prof + p_par) / ETA


def energy_from_speed_profile(speed: np.ndarray, dt: float) -> float:
    """Energía [Wh] integrando la potencia sobre un perfil de velocidad."""
    return float(np.sum(power_electrical(speed)) * dt / 3600.0)


def energy_from_rollout(state: np.ndarray, dt: float) -> float:
    """Energía [Wh] a partir del estado simulado, usando las revoluciones REALES.

    Es la medida más fiel disponible: integra la potencia de eje de los cuatro
    rotores tal como los comandó el controlador, así que incluye el gasto de
    rechazar el viento, que el modelo analítico no ve.
    Estado de 18 componentes: las revoluciones son las posiciones 13..16.
    """
    omega = np.asarray(state)[:, 13:17]
    p_shaft = np.sum(KM * np.abs(omega) ** 3, axis=1)
    return float(np.sum(p_shaft / ETA) * dt / 3600.0)


if __name__ == "__main__":
    print(f"P_hover electrico   = {P_ELEC_HOVER:6.2f} W")
    print(f"v_h                 = {V_H:6.2f} m/s")
    print(f"area de disco       = {DISC_AREA:6.4f} m^2  -> R_eq = {ROTOR_R*100:.1f} cm")
    print(f"{'V [m/s]':>8} {'P [W]':>8} {'vs hover':>9} {'mu':>7}")
    for v in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        p = float(power_electrical(v)[0])
        mu = v / (OMEGA_HOVER * ROTOR_R)
        print(f"{v:8.1f} {p:8.2f} {100*(p/P_ELEC_HOVER-1):+8.1f}% {mu:7.3f}")
