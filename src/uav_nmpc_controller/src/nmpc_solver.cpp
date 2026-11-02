#include "uav_nmpc_controller/nmpc_solver.hpp"

// C generado por acados desde nmpc_model (generate_acados_c_code.py).
// Este wrapper SOLO enlaza ese C — la formulación del OCP no se duplica aquí.
#include "acados_solver_uav_nmpc_18states.h"
#include "acados_sim_solver_uav_nmpc_18states.h"

#include <cmath>
#include <cstdio>

namespace uav {

namespace {
inline uav_nmpc_18states_solver_capsule* cap(void* p) {
  return static_cast<uav_nmpc_18states_solver_capsule*>(p);
}
inline uav_nmpc_18states_sim_solver_capsule* scap(void* p) {
  return static_cast<uav_nmpc_18states_sim_solver_capsule*>(p);
}
}  // namespace

static_assert(NmpcSolver::kNx == UAV_NMPC_18STATES_NX,  "NX mismatch con acados");
static_assert(NmpcSolver::kNu == UAV_NMPC_18STATES_NU,  "NU mismatch con acados");
static_assert(NmpcSolver::kNp == UAV_NMPC_18STATES_NP,  "NP mismatch con acados");

NmpcSolver::~NmpcSolver() {
  if (capsule_) {
    uav_nmpc_18states_acados_free(cap(capsule_));
    uav_nmpc_18states_acados_free_capsule(cap(capsule_));
    capsule_ = nullptr;
  }
  if (sim_capsule_) {
    uav_nmpc_18states_acados_sim_free(scap(sim_capsule_));
    uav_nmpc_18states_acados_sim_solver_free_capsule(scap(sim_capsule_));
    sim_capsule_ = nullptr;
  }
}

bool NmpcSolver::init() {
  capsule_ = uav_nmpc_18states_acados_create_capsule();
  if (!capsule_) return false;
  const int status = uav_nmpc_18states_acados_create(cap(capsule_));
  if (status != 0) {
    std::fprintf(stderr, "[NmpcSolver] acados_create fallo, status=%d\n", status);
    uav_nmpc_18states_acados_free_capsule(cap(capsule_));
    capsule_ = nullptr;
    return false;
  }
  // Integrador exportado (mismo modelo) para la compensación de retardo.
  sim_capsule_ = uav_nmpc_18states_acados_sim_solver_create_capsule();
  if (!sim_capsule_ ||
      uav_nmpc_18states_acados_sim_create(scap(sim_capsule_)) != 0) {
    std::fprintf(stderr, "[NmpcSolver] sim_create fallo\n");
    sim_capsule_ = nullptr;   // predict() devolverá false; el nodo lo maneja
  }
  return true;
}

bool NmpcSolver::predict(std::array<double, kNx>&       x,
                         const std::array<double, kNu>& u,
                         double                         dt_s) {
  if (!sim_capsule_ || dt_s <= 0.0) return false;
  auto* sc = scap(sim_capsule_);
  sim_config* cfg = uav_nmpc_18states_acados_get_sim_config(sc);
  void* dims = uav_nmpc_18states_acados_get_sim_dims(sc);
  sim_in* in = uav_nmpc_18states_acados_get_sim_in(sc);
  sim_out* out = uav_nmpc_18states_acados_get_sim_out(sc);
  sim_in_set(cfg, dims, in, "T", &dt_s);   // horizonte del paso de integración
  sim_in_set(cfg, dims, in, "x", x.data());
  sim_in_set(cfg, dims, in, "u", const_cast<double*>(u.data()));
  if (uav_nmpc_18states_acados_sim_solve(sc) != 0) return false;
  sim_out_get(cfg, dims, out, "xn", x.data());
  // renormalizar el cuaternión (higiene numérica tras integrar)
  const double n = std::sqrt(x[6]*x[6] + x[7]*x[7] + x[8]*x[8] + x[9]*x[9]);
  if (n > 1e-9) { x[6] /= n; x[7] /= n; x[8] /= n; x[9] /= n; }
  return true;
}

void NmpcSolver::reset() {
  if (capsule_) {
    // 1 => resetear también la memoria del QP (HPIPM)
    uav_nmpc_18states_acados_reset(cap(capsule_), 1);
  }
}

bool NmpcSolver::solve(const std::array<double, kNx>& x0,
                       const std::array<double, kNp>& p_ref,
                       double                          preview_offset_s,
                       std::array<double, kNu>&       u0) {
  auto* c = cap(capsule_);
  if (!c) return false;

  ocp_nlp_config* cfg  = uav_nmpc_18states_acados_get_nlp_config(c);
  ocp_nlp_dims*   dims = uav_nmpc_18states_acados_get_nlp_dims(c);
  ocp_nlp_in*     in   = uav_nmpc_18states_acados_get_nlp_in(c);
  ocp_nlp_out*    out  = uav_nmpc_18states_acados_get_nlp_out(c);
  ocp_nlp_solver* sol  = uav_nmpc_18states_acados_get_nlp_solver(c);

  // Estado actual como restricción de igualdad en la etapa 0 (lbx = ubx = x0).
  // NBX0 == NX == 18 (todo el estado está acotado en la etapa 0).
  ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "lbx",
                                const_cast<double*>(x0.data()));
  ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "ubx",
                                const_cast<double*>(x0.data()));

  // Referencia online p = [p_ref, v_ref, q_ref, w_ref] en las N+1 etapas,
  // con PREVIEW lineal: la posición de referencia avanza v_ref*k*dt por etapa,
  // ADELANTADO preview_offset_s (compensación de retardo: el u de esta solve
  // actúa en t+retardo, así que la etapa k corresponde a t+retardo+k*dt).
  // Sin preview, una referencia móvil deja lag ~|v_ref|*T_horizon.
  // dt del OCP = 0.01 s (DT en generate_acados_c_code.py; tf=0.5, N=50).
  constexpr double kDtOcp = 0.01;
  std::array<double, kNp> p_stage = p_ref;
  for (int stage = 0; stage <= UAV_NMPC_18STATES_N; ++stage) {
    const double tau = preview_offset_s + stage * kDtOcp;
    p_stage[0] = p_ref[0] + p_ref[3] * tau;
    p_stage[1] = p_ref[1] + p_ref[4] * tau;
    p_stage[2] = p_ref[2] + p_ref[5] * tau;
    uav_nmpc_18states_acados_update_params(c, stage, p_stage.data(), kNp);
  }

  const int status = uav_nmpc_18states_acados_solve(c);

  double elapsed = 0.0;
  ocp_nlp_get(sol, "time_tot", &elapsed);
  last_solve_ms_ = elapsed * 1000.0;

  // u óptimo de la etapa 0 (se lee aunque status != 0: SQP_RTI entrega
  // la mejor iteración disponible; el llamador decide qué hacer con status).
  ocp_nlp_out_get(cfg, dims, out, 0, "u", u0.data());

  return status == 0;
}

}  // namespace uav
