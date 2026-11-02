#!/usr/bin/env python3
"""make_numbers.py — every number quoted in main.tex, from data/*.json.

Writes numbers.tex (one \\newcommand per value) and tab_coverage.tex and
tab_clearance.tex. Nothing in the manuscript is typed by hand.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
DATA = os.path.join(PAPER, "data")
M: dict[str, str] = {}


def mac(name, value, fmt="{:.1f}"):
    M[name] = fmt.format(value) if not isinstance(value, str) else value


def load(n):
    return json.load(open(os.path.join(DATA, n), encoding="utf-8"))


pct = lambda x: 100.0 * x

# ------------------------------------------------ Table I, corrected planner
cov = load("coverage_study.json")
pub = cov["published"]
f, e, g = pub["free"], pub["exclusion_exact"], pub["exclusion_greedy"]
mac("pubCells", e["cells"], "{:d}")
mac("pubdL", pct(e["path_m"] / f["path_m"] - 1))
mac("pubdE", pct(e["energy_wh"] / f["energy_wh"] - 1))
mac("pubdEarea", pct(e["energy_per_area_mwh_m2"] / f["energy_per_area_mwh_m2"] - 1))
mac("pubdT", pct(e["time_s"] / f["time_s"] - 1))
mac("pubAreaLoss", pct(1 - e["free_area_m2"] / f["free_area_m2"]))
mac("pubGreedyEqual", "identical" if abs(g["energy_wh"] / e["energy_wh"] - 1) < 1e-6 else "different")

# submitted (degenerate) values, from the deposited metrics of the submission
sub = load("metrics.json")["coverage"]
mac("subCells", sub["exclusion_exact"]["cells"], "{:.0f}")
mac("subdL", pct(sub["exclusion_exact"]["path_length_m"] / sub["free"]["path_length_m"] - 1))
mac("subdE", pct(sub["exclusion_exact"]["energy_fwd_wh"] / sub["free"]["energy_fwd_wh"] - 1))
# value printed in the submitted manuscript (22.7 %); subdE is the same
# configuration recomputed with the submitted planner (rounds to 22.6 %)
mac("subdEprinted", 22.7, "{:.1f}")
mac("subTransitCut", pct(1 - sub["exclusion_exact"]["transit_length_m"] / sub["exclusion_greedy"]["transit_length_m"]), "{:.0f}")

# obstacle-shift sweep
sh = cov["shift"]
s = sh["summary"]
mac("shiftN", len(sh["rows"]), "{:d}")
mac("shiftdEmin", pct(s["dE"]["min"])); mac("shiftdEmax", pct(s["dE"]["max"]))
mac("shiftdEmed", pct(s["dE"]["median"]))
mac("shiftdLmin", pct(s["dL"]["min"])); mac("shiftdLmax", pct(s["dL"]["max"]))
mac("shiftShareShorter", pct(s["share_dL_negative"]), "{:.0f}")
mac("shiftGreedyMax", pct(sh["greedy_over_exact"]["max"]))
# Rows fall on inflated obstacle edges only at dy = 0 and dy = 1.0 m (edges at
# -9.5+dy, -3.5+dy, -0.5+dy, -4.5+dy against rows at -12.5+3k); report the
# generic positions separately from those two degenerate ones.
gen = [r for r in sh["rows"] if min(abs(r["dy"]), abs(r["dy"] - 1.0)) > 1e-9]
gE = np.array([r["exact"]["energy_wh"] / f["energy_wh"] - 1 for r in gen])
gL = np.array([r["exact"]["path_m"] / f["path_m"] - 1 for r in gen])
mac("genN", len(gen), "{:d}")
mac("gendEmin", pct(gE.min())); mac("gendEmax", pct(gE.max())); mac("gendEmed", pct(np.median(gE)))
mac("gendLmin", pct(gL.min())); mac("gendLmax", pct(gL.max()))
mac("genShorter", pct(np.mean(gL < 0)), "{:.0f}")
edge1 = [r for r in sh["rows"] if abs(r["dy"] - 1.0) < 1e-9][0]
mac("edgeOnedE", pct(edge1["exact"]["energy_wh"] / f["energy_wh"] - 1))
subshift = load("coverage_shift_submitted_planner.json")
n_spike = sum(r["dE"] > 0.2 for r in subshift)
mac("subShiftSpikeN", n_spike, "{:d}"); mac("subShiftN", len(subshift), "{:d}")

# random fields
rd = cov["random"]; r = rd["summary"]
mac("rndN", rd["n_plans"], "{:d}"); mac("rndFields", rd["n_plans"] // 3, "{:d}")
mac("rndRefused", rd["n_refused"], "{:d}")
mac("rnddLmed", pct(r["dL"]["median"])); mac("rnddLlo", pct(r["dL"]["p25"])); mac("rnddLhi", pct(r["dL"]["p75"]))
mac("rnddEmed", pct(r["dE"]["median"])); mac("rnddElo", pct(r["dE"]["p25"])); mac("rnddEhi", pct(r["dE"]["p75"]))
mac("rnddEfive", pct(r["dE"]["p05"])); mac("rnddEninetyfive", pct(r["dE"]["p95"]))
mac("rndRatioMed", r["dE_over_dL"]["median"], "{:.2f}")
mac("rndRatioLo", r["dE_over_dL"]["p25"], "{:.2f}"); mac("rndRatioHi", r["dE_over_dL"]["p75"], "{:.2f}")
okr = [x for x in rd["rows"] if "error" not in x]
_dL = np.array([x["exact"]["path_m"] / x["free"]["path_m"] - 1 for x in okr])
_dE = np.array([x["exact"]["energy_wh"] / x["free"]["energy_wh"] - 1 for x in okr])
# median dE/dL by number of exclusion zones (plans with |dL| > 0.1 %)
for _k, _w in [(1, "One"), (2, "Two"), (3, "Three")]:
    _m = np.array([x["n_obs"] == _k for x in okr]) & (np.abs(_dL) > 1e-3)
    mac(f"rndRatioK{_w}", float(np.median(_dE[_m] / _dL[_m])), "{:.2f}")
mac("rndShorter", pct(np.mean((_dL < 0) & (_dE > 0))), "{:.0f}")
_m = np.abs(_dL) > 1e-3
mac("rndRatioN", int(_m.sum()), "{:d}"); mac("rndRatioNeg", int(np.sum(_m & (_dE / np.where(_m, _dL, 1) < 0))), "{:d}")
mac("rndRatioOfMedians", np.median(_dE) / np.median(_dL), "{:.2f}")
mac("rndTotdE", pct(r["ratio_of_totals_energy"])); mac("rndTotdL", pct(r["ratio_of_totals_length"]))
mac("rndCovN", rd["n_coverage_matched"], "{:d}")
mac("rndCovdEmed", pct(rd["summary_coverage_matched"]["dE"]["median"]))
mac("rndGreedyMed", pct(rd["greedy_over_exact"]["median"]))
mac("rndGreedyOne", pct(rd["share_greedy_worse_1pct"]), "{:.0f}")
mac("rndGreedyFive", pct(rd["share_greedy_worse_5pct"]), "{:.0f}")
mac("rndGreedyBest", pct(-rd["greedy_over_exact"]["min"]))
mac("rndInvade", rd["inside_inflated_plans"], "{:d}")

# ------------------------------------------------ energy across controllers
mc = load("mc_main.json")
E = {(x["controller"], x["sigma"], x["seed"]): x["energy_wh"] for x in mc}
keys = [(sg, k) for (c, sg, k) in E if c == "nmpc" and sg > 0]
for a, b, tag in [("nmpc", "pid", "NP"), ("nmpc", "lqr", "NL")]:
    d = np.array([E[(a, sg, k)] - E[(b, sg, k)] for sg, k in keys])
    rel = np.array([E[(a, sg, k)] / E[(b, sg, k)] - 1 for sg, k in keys])
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    mac(f"en{tag}", 1e3 * d.mean()); mac(f"en{tag}lo", 1e3 * ci[0]); mac(f"en{tag}hi", 1e3 * ci[1])
    mac(f"en{tag}rel", pct(rel.mean()), "{:.2f}")
    pv = stats.ttest_1samp(d, 0).pvalue
    mac(f"en{tag}p", "$p<10^{-6}$" if pv < 1e-6 else f"$p={pv:.2f}$")
mac("enPairs", len(keys), "{:d}")

# ------------------------------------------------ solve times (re-run, desktop, 4-way parallel)
rr = [x for x in load("mc_main_rerun.json") if x["controller"] == "nmpc"]
mac("solveMeanLo", min(x["solve_mean_ms"] for x in rr), "{:.2f}")
mac("solveMeanHi", max(x["solve_mean_ms"] for x in rr), "{:.2f}")
mac("solvePnnMed", float(np.median([x["solve_p99_ms"] for x in rr])), "{:.2f}")
mac("solveMax", max(x["solve_max_ms"] for x in rr), "{:.1f}")
mac("solveOverBudget", sum(x["solve_max_ms"] > 10.0 for x in rr), "{:d}")
mac("solveEpisodes", len(rr), "{:d}")
# original Table III campaign (mc_main.json), run under heavier parallel load
ro = [x for x in load("mc_main.json") if x["controller"] == "nmpc"]
mac("solveOrigOver", sum(x["solve_max_ms"] > 10.0 for x in ro), "{:d}")
mac("solveOrigMeanLo", min(x["solve_mean_ms"] for x in ro), "{:.2f}")
mac("solveOrigMeanHi", max(x["solve_mean_ms"] for x in ro), "{:.2f}")

# ------------------------------------------------ fairness ablation
fc = defaultdict(list)
for x in load("mc_fair_controllers.json"):
    fc[(x["controller"], x["sigma"])].append(x)
for c, tag in [("pid", "Pid"), ("pid_ff", "PidFf"), ("lqr", "Lqr"), ("lqr_ff", "LqrFf"),
               ("nmpc", "Nmpc"), ("nmpc_extrap", "NmpcEx")]:
    mac(f"fair{tag}", np.mean([x["rmse_m"] for x in fc[(c, 0.55)]]), "{:.3f}")
    mac(f"fair{tag}In", sum(x["steps_inside_raw"] > 0 for s_ in (0.0, 0.55, 0.8) for x in fc[(c, s_)]), "{:d}")
    mac(f"fair{tag}Clr", min(x["min_clearance_m"] for s_ in (0.0, 0.55, 0.8) for x in fc[(c, s_)]), "{:.3f}")
_r = lambda c, s_: np.mean([x["rmse_m"] for x in fc[(c, s_)]])
_e = lambda c, s_: np.mean([x["energy_wh"] for x in fc[(c, s_)]])
_share = [(_r("nmpc_extrap", s_) - _r("nmpc", s_)) / (min(_r("pid", s_), _r("lqr", s_)) - _r("nmpc", s_))
          for s_ in (0.55, 0.8)]
mac("fairPreviewLo", pct(min(_share)), "{:.0f}"); mac("fairPreviewHi", pct(max(_share)), "{:.0f}")
mac("fairNmpcExEnergy", pct(np.mean([_e("nmpc_extrap", s_) / _e("nmpc", s_) - 1 for s_ in (0.0, 0.55, 0.8)])))

# ------------------------------------------------ delay sweep
for fn, tag in [("mc_delay.json", "Sub"), ("mc_delay_rerun.json", "Re")]:
    R = load(fn)
    k7 = [x for x in R if abs(x["delay_ms"] - 7.0) < 1e-9]
    mac(f"divSeven{tag}", sum(x["diverged"] for x in k7), "{:d}")
dep = {(x["delay_ms"], x["seed"]): x["diverged"] for x in load("mc_delay.json")}
same7 = sum(dep[(7.0, x["seed"])] == x["diverged"] for x in load("mc_delay_rerun.json") if x["delay_ms"] == 7.0)
mac("divSevenSame", same7, "{:d}")

# ------------------------------------------------ closed loop on the exclusion plan
ex = load("mc_exclusions.json")
free = {(x["controller"], x["sigma"], x["seed"]): x for x in load("mc_main_rerun.json")}
grp = defaultdict(list)
for x in ex:
    if x["ordering"] == "exact":
        grp[(x["controller"], x["sigma"])].append(x)
# executed vs planned only without disturbance: with sigma > 0 the persistent
# bias has a vertical component that moves an episode's energy by several
# percent either way, which is why the comparisons below are paired by seed
ratios = [x["energy_wh"] / x["energy_plan_wh"] for (c, sg), v in grp.items() if sg == 0.0 for x in v]
mac("execOverPlanLo", pct(min(ratios) - 1)); mac("execOverPlanHi", pct(max(ratios) - 1))
dEx = [x["energy_wh"] / free[(x["controller"], x["sigma"], x["seed"])]["energy_wh"] - 1
       for v in grp.values() for x in v]
mac("execdElo", pct(min(dEx))); mac("execdEhi", pct(max(dEx)))
mac("exclEpisodes", sum(len(v) for v in grp.values()), "{:d}")
rows = []
names = {"pid": "PID", "lqr": "LQR", "nmpc": "NMPC"}
for c in ("pid", "lqr", "nmpc"):
    for sg in (0.0, 0.55, 0.8):
        v = grp[(c, sg)]
        clr = np.array([x["min_clearance_m"] for x in v])
        inside = sum(x["steps_inside_raw"] > 0 for x in v)
        rows.append(f"{names[c]} & {sg:.2f} & {np.mean([x['rmse_m'] for x in v]):.3f} & "
                    f"{clr.min():.3f} & {np.median(clr):.3f} & {sum(clr < 0.5)}/{len(v)} & {inside}/{len(v)} \\\\")
head = ["\\begin{tabular}{lcccccc}", "\\toprule",
        " & $\\sigma$ & RMSE & min & median & $<0.5$~m & inside \\\\", "\\midrule"]
open(os.path.join(PAPER, "tab_clearance.tex"), "w").write(
    "\n".join(head + rows + ["\\bottomrule", "\\end{tabular}"]) + "\n")

# Table I (corrected planner)
rowsI = [("Cells", "cells", "{:d}"), ("Path length [m]", "path_m", "{:.1f}"),
         ("\\quad transit [m]", "transit_m", "{:.1f}"), ("Mission time [s]", "time_s", "{:.1f}"),
         ("Free area [m$^2$]", "free_area_m2", "{:.1f}"), ("Energy [Wh]", "energy_wh", "{:.3f}"),
         ("Energy/area [mWh\\,m$^{-2}$]", "energy_per_area_mwh_m2", "{:.2f}"),
         ("Field coverage [\\%]", "coverage_field", "pct"), ("Accessible coverage [\\%]", "coverage_accessible", "pct")]
fmt = lambda v, f_: f"{100 * v:.1f}" if f_ == "pct" else f_.format(v)
tab = ["\\begin{tabular}{lccc}", "\\toprule",
       "& \\textbf{No exclusions} & \\textbf{Exclusions} & \\textbf{Exclusions} \\\\",
       "& & \\textbf{(exact)} & \\textbf{(greedy)} \\\\", "\\midrule"]
for label, key, f_ in rowsI:
    tv = "---" if key == "transit_m" and f[key] == 0 else fmt(f[key], f_)
    tab.append(f"{label} & {tv} & {fmt(e[key], f_)} & {fmt(g[key], f_)} \\\\")
tab += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(PAPER, "tab_coverage.tex"), "w").write("\n".join(tab) + "\n")

lines = ["% Generated by scripts/make_numbers.py -- do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
open(os.path.join(PAPER, "numbers.tex"), "w").write("\n".join(lines) + "\n")
print(f"wrote numbers.tex ({len(M)} macros) and tab_clearance.tex")
for k, v in sorted(M.items()):
    print(f"  {k} = {v}")
