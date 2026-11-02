#!/usr/bin/env python3
"""mc_experiments.py — Campaña Monte Carlo sobre el barrido de cobertura.

Responde a las tres objeciones más duras de la revisión: N=1, ausencia de
baselines y ausencia de barrido de retardo. Lo hace SIN Gazebo: cierra el lazo
sobre la misma planta de 18 estados del MIL, con los mismos controladores ya
validados (PID en cascada, LQR de hover, NMPC exportado por acados). Gazebo
aporta las cámaras y la geometría del cultivo, que aquí no hacen falta.

Diseñado para reutilizarse: añadir un controlador, un nivel de viento o una
variante de planificador es una entrada más en las listas de barrido, no un
script nuevo.

Uso:
  python3 mc_experiments.py --controllers pid,lqr,nmpc --sigmas 0,0.3,0.55,0.8 \
                            --seeds 20 --jobs 12
  python3 mc_experiments.py --delay-sweep --seeds 10 --jobs 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
MIL = os.path.join(ROOT, "nmpc_model")
sys.path.insert(0, MIL)
sys.path.insert(0, os.path.join(ROOT, "src", "mission_planner"))
sys.path.insert(0, HERE)

os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))
os.environ["LD_LIBRARY_PATH"] = (os.path.join(os.environ["ACADOS_SOURCE_DIR"], "lib")
                                 + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
# Generated solver code goes here, never into the source tree (build/ is
# ignored). A fixed name keeps the C symbols stable across acados versions.
BUILD_DIR = os.environ.get("ACADOS_BUILD_DIR", os.path.join(ROOT, "build", "acados"))

DT = 0.01                      # paso de control [s] = paso del OCP
N_HORIZON = 50
OUT = os.path.join(PAPER, "data")


# --------------------------------------------------------------- referencia
def build_reference(spacing=3.0, speed=2.0, altitude=7.0, obstacles=None,
                    ordering="exact"):
    """Barrido boustrophedon -> traj_fn(t) = (p, v, q) muestreado a DT."""
    from mission_planner.coverage_planner import CoveragePlanner
    field = [(0.0, 0.0), (20.0, 0.0), (20.0, -14.0), (0.0, -14.0)]
    pl = CoveragePlanner(field, altitude, speed, track_spacing_m=spacing,
                         obstacles_xy=obstacles, obstacle_clearance_m=0.5,
                         yaw_mode="fixed", cell_ordering=ordering)
    plan = pl.plan()
    samples = pl.time_parameterize(plan, dt=DT)
    t = np.array([s[0] for s in samples])
    p = np.array([[s[1], s[2], s[3]] for s in samples])
    v = np.gradient(p, DT, axis=0)
    n = len(t)

    def traj_fn(tq: float):
        k = int(round(tq / DT))
        k = min(max(k, 0), n - 1)          # se mantiene el punto final al acabar
        return p[k], (v[k] if k < n - 1 else np.zeros(3)), np.array([1.0, 0, 0, 0])

    return traj_fn, n, plan.metrics


# ------------------------------------------------------------------- viento
def make_wind_seq(sigma, seed, n):
    """OU de media cero + sesgo persistente por episodio (fuerza [N])."""
    rng = np.random.RandomState(seed)
    bdir = rng.randn(3)
    bdir[2] *= 0.5
    bdir /= (np.linalg.norm(bdir) + 1e-9)
    bias = bdir * (0.6 if sigma > 0 else 0.0)
    w = np.zeros((n, 3))
    cur = np.zeros(3)
    for k in range(n):
        w[k] = cur + bias
        cur = cur - 1.0 * cur * DT + sigma * np.sqrt(DT) * rng.randn(3)
    return w


# ------------------------------------------------------------- controladores
_CACHE = {}


def get_controller(name):
    """Controlador para un episodio.

    PID/LQR se cachean por proceso y se reinician en cada episodio. El NMPC se
    crea de nuevo en cada episodio a partir del código ya compilado en
    BUILD_DIR (~10 ms): reutilizarlo arrastraba el warm-start del episodio
    anterior, y los resultados dependían del orden de ejecución.
    """
    if name in ("pid", "lqr"):
        if name not in _CACHE:
            from linear_baselines import make_baselines
            _CACHE[name] = make_baselines()[name]
        return _CACHE[name]
    if name in ("pid_ff", "lqr_ff"):
        if name not in _CACHE:
            from baselines_ff import CascadePIDFF, HoverLQRFF
            _CACHE[name] = CascadePIDFF() if name == "pid_ff" else HoverLQRFF()
        return _CACHE[name]
    if name in ("nmpc", "nmpc_extrap"):
        from acados_template import AcadosOcpSolver
        import generate_acados_c_code as G
        ocp = G.build_ocp()
        ocp.name = "uav_nmpc_18states"
        ocp.code_gen_options.code_export_directory = BUILD_DIR
        ocp.code_gen_options.json_file = "uav_nmpc_ocp.json"
        return AcadosOcpSolver(ocp, build=False, generate=False, verbose=False)
    raise ValueError(name)


def nmpc_action(solver, x18, t, traj_fn, delta_hover, preview=True):
    """preview=True: the reference over the whole horizon (conference MC).
    preview=False: the current reference extrapolated with its velocity, as
    the ROS 2 node does -- the same information the linear baselines get."""
    p0, v0, q0 = traj_fn(t)
    for k in range(N_HORIZON + 1):
        if preview:
            p_k, v_k, q_k = traj_fn(t + k * DT)
        else:
            p_k, v_k, q_k = p0 + v0 * k * DT, v0, q0
        solver.set(k, "p", np.concatenate([p_k, v_k, q_k, np.zeros(3)]))
    solver.set(0, "lbx", x18)
    solver.set(0, "ubx", x18)
    t0 = time.perf_counter()
    status = solver.solve()
    ms = (time.perf_counter() - t0) * 1e3
    if status != 0:
        return np.full(4, delta_hover), ms, False
    return np.array(solver.get(0, "u")), ms, True


# ---------------------------------------------------------------- un episodio
def run_episode(job):
    (controller, sigma, seed, delay_ms, spacing, obstacles, ordering,
     substeps) = job
    import casadi as ca
    from uav_dynamics_casadi import (build_model_18states, OMEGA_HOVER, V_NOM,
                                     MASS, DELTA_HOVER)
    import power_model as pw

    delta_hover = DELTA_HOVER
    traj_fn, n, plan_metrics = build_reference(spacing, obstacles=obstacles,
                                               ordering=ordering)
    # energy the planning model assigns to the same reference (Table I model),
    # so planned and executed energy are compared on identical plans
    p_ref = np.array([traj_fn(k * DT)[0] for k in range(n)])
    speed_ref = np.linalg.norm(np.gradient(p_ref[:, :2], DT, axis=0), axis=1)
    energy_plan_wh = float(pw.energy_from_speed_profile(speed_ref, DT))
    wind = make_wind_seq(sigma, seed, n)

    x_sym, u_sym, xdot, _ = build_model_18states()
    f18 = ca.Function("f18", [x_sym, u_sym], [xdot])

    def rk4(x, u, w, h=DT):
        acc = np.zeros(18)
        acc[3:6] = w / MASS

        def fd(xx):
            return np.array(f18(ca.DM(xx), ca.DM(u))).flatten() + acc
        k1 = fd(x); k2 = fd(x + h / 2 * k1)
        k3 = fd(x + h / 2 * k2); k4 = fd(x + h * k3)
        xn = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[6:10] /= np.linalg.norm(xn[6:10])
        return xn

    ctrl = get_controller(controller)
    if controller in ("pid", "lqr", "pid_ff", "lqr_ff"):
        ctrl.reset()

    p0, v0, _ = traj_fn(0.0)
    x = np.concatenate([p0, v0, [1.0, 0, 0, 0], np.zeros(3),
                        np.full(4, OMEGA_HOVER), [V_NOM]]).astype(float)

    err = np.zeros(n)
    state = np.zeros((n, 18))
    solve_ms = []
    n_fail = 0
    # La planta se integra `substeps` veces más fino que el lazo de control.
    # Sin esto, el búfer de comandos solo puede representar retardos múltiplos
    # del periodo de control (10 ms), y el margen de estabilidad —que está por
    # debajo de ese valor— queda por completo entre dos muestras.
    dt_sub = DT / substeps
    buf = deque([np.full(4, delta_hover)] * int(round(delay_ms / 1e3 / dt_sub)),
                maxlen=None)
    diverged = False

    for k in range(n):
        t = k * DT
        p_ref, v_ref, _ = traj_fn(t)
        if controller in ("nmpc", "nmpc_extrap"):
            u, ms, ok = nmpc_action(ctrl, x, t, traj_fn, delta_hover,
                                    preview=(controller == "nmpc"))
            solve_ms.append(ms)
            n_fail += (not ok)
        elif controller in ("pid_ff", "lqr_ff"):
            a_ref = (np.asarray(traj_fn(t + DT)[1]) - np.asarray(traj_fn(t - DT)[1])) / (2 * DT)
            u = ctrl.compute(x, np.asarray(p_ref), np.asarray(v_ref), a_ref)
        else:
            u = ctrl.compute(x, np.asarray(p_ref), np.asarray(v_ref))
        u = np.clip(np.asarray(u, float), 0.0, 1.0)

        err[k] = np.linalg.norm(x[0:3] - p_ref)
        state[k] = x
        for _ in range(substeps):
            if buf.maxlen != 0 and len(buf) > 0:
                buf.append(u)
                u_applied = buf.popleft()
            else:
                u_applied = u
            x = rk4(x, u_applied, wind[k], dt_sub)
        if not np.all(np.isfinite(x)) or abs(x[2]) > 100 or err[k] > 25.0:
            diverged = True
            err[k:] = err[k]
            state[k:] = state[k]
            break

    res = {
        "controller": controller, "sigma": sigma, "seed": seed,
        "delay_ms": float(delay_ms), "substeps": substeps, "spacing": spacing,
        "obstacles": bool(obstacles), "ordering": ordering,
        "diverged": bool(diverged),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "e_p50_m": float(np.percentile(err, 50)),
        "e_p95_m": float(np.percentile(err, 95)),
        "e_max_m": float(np.max(err)),
        "energy_wh": pw.energy_from_rollout(state, DT),
        "energy_plan_wh": energy_plan_wh,
        "mission_s": n * DT,
        "path_length_m": plan_metrics.get("path_length_m"),
        "n_cells": plan_metrics.get("cells"),
        "transit_m": plan_metrics.get("transit_length_m"),
    }
    traj_dir = os.environ.get("MC_SAVE_TRAJ")
    if traj_dir:
        # executed track, reference and controls for post-hoc analysis
        os.makedirs(traj_dir, exist_ok=True)
        p_ref_all = np.array([traj_fn(k * DT)[0] for k in range(n)])
        np.savez_compressed(
            os.path.join(traj_dir, f"{controller}_s{sigma}_d{delay_ms}_{ordering}_{seed}.npz"),
            state=state, p_ref=p_ref_all, err=err, wind=wind, dt=DT)
    if obstacles:
        # clearance of the EXECUTED track to the raw no-fly polygons; the plan
        # keeps 0.5 m, the vehicle under wind need not
        import shapely
        from shapely.geometry import Polygon
        pts = shapely.points(state[:, 0], state[:, 1])
        raw = [Polygon(o) for o in obstacles]
        res["min_clearance_m"] = float(min(shapely.distance(pts, o).min() for o in raw))
        res["steps_inside_raw"] = int(sum(shapely.contains_xy(o, state[:, 0], state[:, 1]).sum()
                                          for o in raw))
    if solve_ms:
        s = np.array(solve_ms)
        res.update(solve_mean_ms=float(s.mean()), solve_max_ms=float(s.max()),
                   solve_p99_ms=float(np.percentile(s, 99)),
                   solver_failures=int(n_fail))
    return res


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", default="pid,lqr,nmpc")
    ap.add_argument("--sigmas", default="0,0.3,0.55,0.8")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--delay-sweep", action="store_true")
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--substeps", type=int, default=1,
                    help="submuestreo de la planta (10 => resolucion de 1 ms)")
    ap.add_argument("--orderings", default="exact",
                    help="lista: exact,greedy (orden de celdas)")
    ap.add_argument("--obstacle-shift-mm", type=float, default=0.0,
                    help="desplaza los obstáculos en y; 1 mm rompe la "
                         "coincidencia de bordes con las líneas de barrido")
    ap.add_argument("--out", default="mc_results.json")
    a = ap.parse_args()

    dy = a.obstacle_shift_mm / 1e3
    obstacles = ([[(x, y + dy) for x, y in o]
                  for o in [[(6, -4), (10, -4), (10, -9), (6, -9)],
                            [(14, -1), (17, -1), (17, -4), (14, -4)]]]
                 if a.obstacles else None)

    jobs = []
    if a.delay_sweep:
        for d_ms in (0, 2, 4, 5, 6, 7, 8, 10):
            for seed in range(a.seeds):
                jobs.append(("nmpc", 0.55, 1000 + seed, float(d_ms),
                             a.spacing, obstacles, "exact", a.substeps))
    else:
        for c in a.controllers.split(","):
            for s in [float(v) for v in a.sigmas.split(",")]:
                for o in a.orderings.split(","):
                    for seed in range(a.seeds):
                        jobs.append((c, s, 1000 + seed, 0.0, a.spacing, obstacles,
                                     o, a.substeps))

    path = os.path.join(OUT, a.out)          # an absolute --out wins
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; results are never overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if any(j[0].startswith("nmpc") for j in jobs):
        get_controller("nmpc")                # compile once, before forking

    print(f"{len(jobs)} episodios en {a.jobs} procesos")
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f} s)", flush=True)

    results.sort(key=lambda r: (r["controller"], r["sigma"], r["delay_ms"],
                                r["ordering"], r["seed"]))
    json.dump(results, open(path, "w"), indent=1)
    import subprocess
    git = lambda *c: subprocess.run(["git", "-C", ROOT, *c], capture_output=True,
                                    text=True).stdout.strip()
    acados_sha = os.path.join(os.environ["ACADOS_SOURCE_DIR"], "lib", "git_commit_hash")
    manifest = {"argv": sys.argv, "jobs": len(jobs), "git_sha": git("rev-parse", "HEAD"),
                "git_status": git("status", "--porcelain"),
                "acados_commit": (open(acados_sha).read().strip()
                                  if os.path.exists(acados_sha) else None),
                "build_dir": BUILD_DIR, "wall_s": time.time() - t0}
    json.dump(manifest, open(os.path.splitext(path)[0] + ".manifest.json", "w"), indent=1)
    print(f"-> {path}  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
