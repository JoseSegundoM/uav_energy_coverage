#pragma once
#include <array>
// Interfaz del solver NMPC. La implementación enlaza el C generado por acados
// (modelo uav_nmpc_18states exportado desde nmpc_model — NO se reimplementa).
namespace uav {

class NmpcSolver {
public:
  // Dimensiones del OCP exportado (ver acados_solver_uav_nmpc_18states.h)
  static constexpr int kNx = 18;  // [p(3), v(3), q_wxyz(4), w(3), Omega(4), V_bat]
  static constexpr int kNu = 4;   // PWM normalizado delta_i in [0,1]
  static constexpr int kNp = 13;  // referencia online: [p_ref(3), v_ref(3), q_ref_wxyz(4), w_ref(3)]

  ~NmpcSolver();

  // Crea la capsule de acados. false si el create devuelve estado != 0.
  bool init();

  // x0: estado actual (orden EXACTO del plant de acados).
  // p_ref: referencia online (se fija en las N+1 etapas del horizonte).
  // preview_offset_s: adelanto temporal del preview [s] — con compensación de
  //        retardo, la referencia de la etapa k corresponde al instante
  //        t + retardo + k*dt (el u resuelto ahora actúa tras el retardo).
  // u0: salida — PWM óptimo de la etapa 0.
  // Devuelve false si el solver retorna estado != 0 (u0 igual se escribe).
  bool solve(const std::array<double, kNx>& x0,
             const std::array<double, kNp>& p_ref,
             double                          preview_offset_s,
             std::array<double, kNu>&       u0);

  // Propaga x un paso de dt_s con el integrador EXPORTADO por acados
  // (acados_sim_solver_uav_nmpc_18states — mismo modelo, no se reimplementa).
  // Base de la compensación de retardo: predecir el estado en el que estará
  // la planta cuando el comando recién calculado llegue a aplicarse.
  bool predict(std::array<double, kNx>&       x,
               const std::array<double, kNu>& u,
               double                         dt_s);

  // Resetea la memoria interna del solver (QP + iterados RTI). Llamar tras un
  // status != 0: SQP_RTI no se recupera solo de un QP degenerado.
  void reset();

  // Tiempo del último solve reportado por acados [ms] (verificación dt > solve).
  double lastSolveTimeMs() const { return last_solve_ms_; }

private:
  void* capsule_{nullptr};      // uav_nmpc_18states_solver_capsule* (opaco)
  void* sim_capsule_{nullptr};  // uav_nmpc_18states_sim_solver_capsule*
  double last_solve_ms_{0.0};
};

}  // namespace uav
