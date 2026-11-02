"""Planning-level geometric sensitivity of the exclusion energy gap (235).
# Provenance: generated with the SUBMITTED planner (uav_agricultural@85eb5ee) as
#   python coverage_shift_submitted_planner.py --part shift --tree repository --out shift_original.json
# from the HRFEST-2026 audit workspace; kept to regenerate Fig. 2(c). Do not rerun
# against the corrected planner: it reproduces the degenerate decomposition only
# with the submitted code.
--part shift : obstacles shifted in y over [-1.5, 1.5] m, original vs fixed planner.
--part random: random fields/exclusions/sweep angles, fixed planner.
Energy is the paper's model (eval_metrics.py:163-169 replica)."""
import argparse, importlib, json, math, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True)
ap.add_argument('--tree', default='dev'); ap.add_argument('--n', type=int, default=200)
ap.add_argument('--seed', type=int, default=20261022); ap.add_argument('--out', required=True)
a = ap.parse_args()
# Checkout of the submitted planner to compare against (uav_agricultural@85eb5ee
# for the paper's dashed curve); set SUBMITTED_PLANNER_REPO to its path.
REPO = os.environ.get('SUBMITTED_PLANNER_REPO')
if not REPO:
    sys.exit('set SUBMITTED_PLANNER_REPO to a checkout of the submitted planner')
sys.path.insert(0, f'{REPO}/src/mission_planner'); sys.path.insert(0, f'{REPO}/Hhrfest_paper/scripts')
from mission_planner.coverage_planner import CoveragePlanner
try:
    from mission_planner.coverage_planner import PlanningError
except ImportError:
    PlanningError = None
import power_model as pw
from shapely.geometry import Polygon, LineString

def energy(pl, plan, dt=0.05):
    s = pl.time_parameterize(plan, dt=dt)
    xy = np.array([[t, x, y] for t, x, y, _, _ in s])
    v = np.gradient(xy[:, 1:3], dt, axis=0)
    return float(pw.energy_from_speed_profile(np.hypot(v[:, 0], v[:, 1]), dt)), float(xy[-1, 0])

class Fixed(CoveragePlanner):
    def __init__(self, *x, ang=None, **k):
        self._ang = ang; super().__init__(*x, **k)
    def _optimal_sweep_angle(self):
        return super()._optimal_sweep_angle() if self._ang is None else self._ang

def run(field, obs, ang=None, ordering='exact'):
    pl = Fixed(field, 7.0, 2.0, ang=ang, track_spacing_m=3.0, obstacles_xy=obs or None,
               obstacle_clearance_m=0.5, cell_ordering=ordering)
    plan = pl.plan(); E, T = energy(pl, plan)
    w = plan.waypoints_xy; inv = 0.0
    for p, q in zip(w, w[1:]):
        if math.dist(p, q) > 1e-9:
            inv += sum(LineString([p, q]).intersection(o.buffer(-1e-6)).length for o in pl.obstacles)  # inflated
    return dict(L=plan.metrics['path_length_m'], cells=plan.metrics.get('cells', 1.0), E=E, T=T,
                A=pl.poly.area, cov=plan.metrics.get('coverage_accessible'), invade_raw_m=inv)

FIELD = [(0, 0), (20, 0), (20, -14), (0, -14)]
OBST = [[(6, -4), (10, -4), (10, -9), (6, -9)], [(14, -1), (17, -1), (17, -4), (14, -4)]]
rows = []
if a.part == 'shift':
    free = run(FIELD, None)
    grid = np.unique(np.r_[np.round(np.arange(-1.5, 1.5001, 0.01), 6), [-1e-3, 1e-3, -1e-6, 1e-6]])
    for dy in grid:
        obs = [[(x, y + dy) for x, y in o] for o in OBST]
        r = {}
        for o in ('exact', 'greedy'):
            r[o] = run(FIELD, obs, ordering=o)
        rows.append(dict(dy=float(dy), cells=r['exact']['cells'], dL=r['exact']['L'] / free['L'] - 1,
                         dE=r['exact']['E'] / free['E'] - 1, dEgreedy=r['greedy']['E'] / free['E'] - 1,
                         dEarea=(r['exact']['E'] / r['exact']['A']) / (free['E'] / free['A']) - 1,
                         invade_raw_m=r['exact']['invade_raw_m'] + r['greedy']['invade_raw_m']))
else:
    rng = np.random.default_rng(a.seed)
    def rect(cx, cy, w, h, th):
        c, s = math.cos(th), math.sin(th)
        return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]]
    for i in range(a.n):
        W = rng.uniform(15, 60); H = W / rng.uniform(1, 3)
        field = [(0, 0), (W, 0), (W, -H), (0, -H)]
        obs = []
        for _ in range(int(rng.integers(1, 4))):
            for _try in range(50):
                w, h = rng.uniform(2, 0.25 * W), rng.uniform(2, 0.35 * H)
                cx, cy = rng.uniform(0.2 * W, 0.8 * W), -rng.uniform(0.2 * H, 0.8 * H)
                o = rect(cx, cy, w, h, rng.uniform(0, math.pi) if rng.random() < 0.5 else 0.0)
                P = Polygon(o)
                if Polygon(field).buffer(-1.0).contains(P) and all(P.distance(Polygon(q)) > 1.5 for q in obs):
                    obs.append(o); break
        for ang in [0.0, math.pi / 4, math.pi / 2]:   # Huang angle = 0 for rectangles
            try:
                f = run(field, None, ang); e = run(field, obs, ang, 'exact'); g = run(field, obs, ang, 'greedy')
            except Exception as ex:
                rows.append(dict(i=i, ang=ang, error=type(ex).__name__, msg=str(ex)[:120])); continue
            rows.append(dict(i=i, ang=round(math.degrees(ang)), cov_free=f['cov'], W=W, H=H, n_obs=len(obs),
                             excl_frac=1 - e['A'] / f['A'], cells=e['cells'], dL=e['L'] / f['L'] - 1,
                             dE=e['E'] / f['E'] - 1, dEarea=(e['E'] / e['A']) / (f['E'] / f['A']) - 1,
                             greedy_over_exact=g['E'] / e['E'] - 1, invade_inflated_m=e['invade_raw_m'] + g['invade_raw_m'],
                             cov=e['cov']))
        if i % 20 == 0: print(i, flush=True)
json.dump(rows, open(a.out, 'w'), indent=1)
print('rows', len(rows))
