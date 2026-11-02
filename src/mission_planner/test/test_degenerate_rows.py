"""A scan row on an inflated obstacle edge must not create cells or stops.

In the published field three rows lie exactly on inflated edges. The free
area (a closed set) then meets each of them in collinear pieces that touch,
which the decomposition used to read as a critical point: 9 cells instead of
6 and stops in the middle of straight rows. A row on the edge is flyable at
exactly the clearance, so the plan at the coincidence may legitimately differ
from the plans just above and below it, but it must not be worse than both.
"""
import math

from mission_planner.coverage_planner import CoveragePlanner

FIELD = [(0, 0), (20, 0), (20, -14), (0, -14)]
OBST = [[(6, -4), (10, -4), (10, -9), (6, -9)], [(14, -1), (17, -1), (17, -4), (14, -4)]]


def _plan(dy):
    obs = [[(x, y + dy) for x, y in o] for o in OBST]
    return CoveragePlanner(FIELD, 7.0, 2.0, track_spacing_m=3.0, obstacles_xy=obs,
                           obstacle_clearance_m=0.5).plan()


def test_edge_coincidence_adds_no_cells_or_length():
    at, up, down = _plan(0.0), _plan(1e-6), _plan(-1e-6)
    assert at.metrics["cells"] <= max(up.metrics["cells"], down.metrics["cells"])
    assert at.metrics["path_length_m"] <= max(up.metrics["path_length_m"],
                                              down.metrics["path_length_m"]) + 1e-3


def test_no_stop_in_the_middle_of_a_straight_row():
    w = _plan(0.0).waypoints_xy
    for a, b, c in zip(w, w[1:], w[2:]):
        u = (b[0] - a[0], b[1] - a[1])
        v = (c[0] - b[0], c[1] - b[1])
        if math.hypot(*u) < 1e-9 or math.hypot(*v) < 1e-9:
            continue
        cross = u[0] * v[1] - u[1] * v[0]
        dot = u[0] * v[0] + u[1] * v[1]
        assert not (abs(cross) < 1e-9 and dot > 0), (a, b, c)
