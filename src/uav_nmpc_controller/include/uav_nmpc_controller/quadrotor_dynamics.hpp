#pragma once
// Estado/entrada del plant de acados `uav_nmpc_18states` (fuente de verdad:
// nmpc_model/uav_dynamics_casadi.py — NO duplicar la
// dinámica aquí; este header solo documenta el ORDEN para el puente ROS).
//
//   x = [x, y, z,            posición inercial                    (0..2)
//        dx, dy, dz,         velocidad lineal, marco MUNDO        (3..5)
//        q0, q1, q2, q3,     cuaternión [w,x,y,z], norma 1        (6..9)
//        p, q, r,            velocidad angular body-frame         (10..12)
//        Om1, Om2, Om3, Om4, velocidad de rotores [rad/s]         (13..16)
//        V_bat]              tensión de batería [V]               (17)
//   u = [d1, d2, d3, d4]     PWM normalizado, d_i in [0, 1]
namespace uav {

struct State {
  double p[3];        // posición inercial [m]
  double v[3];        // velocidad lineal, marco mundo [m/s]
  double q[4];        // cuaternión de actitud [w,x,y,z]
  double w[3];        // body rates [rad/s]
  double omega[4];    // velocidad de rotores [rad/s] (hover ~292.0)
  double v_bat;       // tensión de batería [V] (nominal 16.8)
};

struct Input {
  double delta[4];    // PWM normalizado in [0,1] (hover ~0.1947)
};

}  // namespace uav
