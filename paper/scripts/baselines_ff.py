"""baselines_ff.py — PID and LQR baselines with reference acceleration feedforward.

The conference baselines (nmpc_model/linear_baselines.py, left untouched)
act on position and velocity errors only, while the NMPC
sees the whole reference over its horizon. These variants give the linear
controllers the reference information a trajectory tracker normally has, the
reference acceleration a_ref, so that the comparison separates what comes from
the control law from what comes from the information it receives.

  CascadePIDFF  a_des = PID(e_p, e_v) + a_ref
  HoverLQRFF    reference attitude and thrust implied by a_ref
                (theta_ref = a_x/g, phi_ref = -a_y/g, dU1 = m a_z)
"""
from __future__ import annotations

import numpy as np

from linear_baselines import (CascadePID, HoverLQR, DT, G, MASS, quat_to_R,
                             quat_to_euler, wrench_to_delta)


class CascadePIDFF(CascadePID):
    """Same gains and inner loop as CascadePID, plus a_ref in the outer loop."""

    def compute(self, x, p_ref, v_ref, a_ref=np.zeros(3)):
        p = x[0:3]; v = x[3:6]; q = x[6:10]; omega = x[10:13]
        ep = p_ref - p; ev = v_ref - v
        self._int = np.clip(self._int + ep * DT, -1.5, 1.5)
        a_des = (self.Kp_p * ep + self.Kd_p * ev + self.Ki_p * self._int
                 + np.asarray(a_ref, float))
        F_des = MASS * (a_des + np.array([0.0, 0.0, G]))
        zb_des = F_des / (np.linalg.norm(F_des) + 1e-9)
        roll_des = np.clip(np.arcsin(np.clip(-zb_des[1], -1, 1)), -self.tilt_max, self.tilt_max)
        pitch_des = np.clip(np.arctan2(zb_des[0], zb_des[2]), -self.tilt_max, self.tilt_max)
        U1 = max(float(F_des @ quat_to_R(q)[:, 2]), 0.2 * MASS * G)
        roll, pitch, yaw = quat_to_euler(q)
        e_att = np.array([roll_des - roll, pitch_des - pitch, 0.0 - yaw])
        tau = self.I * (self.Kp_att * e_att - self.Kd_att * omega)
        return wrench_to_delta(U1, tau)


class HoverLQRFF(HoverLQR):
    """Same gain matrix as HoverLQR, with the reference state and thrust of a_ref."""

    def compute(self, x, p_ref, v_ref, a_ref=np.zeros(3)):
        a_ref = np.asarray(a_ref, float)
        p = x[0:3]; v = x[3:6]; q = x[6:10]; omega = x[10:13]
        roll, pitch, yaw = quat_to_euler(q)
        st = np.array([p[0], p[1], p[2], v[0], v[1], v[2], roll, pitch, yaw,
                       omega[0], omega[1], omega[2]])
        ref = np.array([p_ref[0], p_ref[1], p_ref[2], v_ref[0], v_ref[1], v_ref[2],
                        -a_ref[1] / G, a_ref[0] / G, 0, 0, 0, 0])
        u = -self.K @ (st - ref)
        U1 = max(MASS * G + MASS * a_ref[2] + u[0], 0.2 * MASS * G)
        return wrench_to_delta(U1, u[1:4])
