"""The emitted plan must not enter a no-fly polygon (inflated by the clearance).

Before the fix, the link between consecutive rows of one cell was a straight
segment never checked against obstacles, and this field (a 40 x 28 m plot with
a rotated exclusion) produced a plan that crossed it.
"""
import math

from shapely.geometry import LineString

from mission_planner.coverage_planner import CoveragePlanner


def _rotated_rect(cx, cy, w, h, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in pts]


FIELD = [(0, 0), (40, 0), (40, -28), (0, -28)]
OBST = [[(8, -6), (14, -6), (14, -13), (8, -13)],
        _rotated_rect(27, -8, 5, 3, 35),
        [(20, -19), (26, -19), (26, -24), (20, -24)]]


def test_plan_stays_out_of_inflated_obstacles():
    for ordering in ("greedy", "exact"):
        pl = CoveragePlanner(FIELD, 7.0, 2.0, track_spacing_m=3.0, obstacles_xy=OBST,
                             obstacle_clearance_m=0.5, cell_ordering=ordering)
        w = pl.plan().waypoints_xy
        for a, b in zip(w, w[1:]):
            if math.dist(a, b) < 1e-9:
                continue
            seg = LineString([a, b])
            for o in pl.obstacles:
                assert seg.intersection(o.buffer(-1e-6)).length < 1e-6, (ordering, a, b)


def test_disconnected_free_area_is_refused_not_crossed():
    """An inflated wall splitting the field leaves no free route between the
    halves; the planner used to emit the straight segment through it."""
    import pytest
    from mission_planner.coverage_planner import PlanningError
    field = [(0, 0), (20, 0), (20, -14), (0, -14)]
    wall = [[(9, -0.5), (11, -0.5), (11, -13.5), (9, -13.5)]]
    with pytest.raises(PlanningError):
        CoveragePlanner(field, 7.0, 2.0, track_spacing_m=3.0, obstacles_xy=wall,
                        obstacle_clearance_m=1.0).plan()
