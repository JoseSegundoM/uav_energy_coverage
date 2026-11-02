# Results run bottom-up: coverage study and Monte Carlo -> numbers and figures -> paper.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
ACADOS ?= $(HOME)/acados

.PHONY: help setup test numbers figures paper verify acados coverage-study monte-carlo

help:
	@echo "setup           virtual environment and dependencies"
	@echo "test            planner regression tests"
	@echo "numbers         paper/numbers.tex and Tables I-II from paper/data/"
	@echo "figures         Figs. 2-4 into paper/figures/"
	@echo "paper           both PDF profiles into paper/ (needs TeX Live)"
	@echo "verify          data checksums; regenerate numbers and Tables I-II and fail if they differ"
	@echo "acados          export the NMPC C code (needs acados, ACADOS=<path>)"
	@echo "coverage-study  geometric study and 600 random plans into build/ (about 1 h)"
	@echo "monte-carlo     360-mission campaign into build/ (needs acados, CPU-hours)"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=src/mission_planner $(PY) -m pytest src/mission_planner/test -q -p no:cacheprovider

numbers:
	$(PY) paper/scripts/make_numbers.py

figures:
	PYTHONPATH=src/mission_planner $(PY) paper/scripts/make_figures.py fig_coverage
	PYTHONPATH=src/mission_planner $(PY) paper/scripts/make_figures.py fig_delay
	$(PY) paper/scripts/fig_clearance.py

paper:
	bash paper/build_profiles.sh

verify:
	cd paper/data && sha256sum -c --quiet SHA256SUMS
	$(PY) paper/scripts/make_numbers.py
	git diff --stat --exit-code paper/numbers.tex paper/tab_coverage.tex paper/tab_clearance.tex

acados:
	ACADOS_SOURCE_DIR=$(ACADOS) LD_LIBRARY_PATH=$(ACADOS)/lib $(PY) nmpc_model/generate_acados_c_code.py

coverage-study:
	$(PY) paper/scripts/coverage_study.py --jobs 3 --out $(abspath build/coverage_study.json)

monte-carlo:
	ACADOS_SOURCE_DIR=$(ACADOS) LD_LIBRARY_PATH=$(ACADOS)/lib $(PY) paper/scripts/mc_experiments.py \
	  --controllers pid,lqr,nmpc --sigmas 0,0.3,0.55,0.8 --seeds 30 --jobs 4 --out $(abspath build/mc_main.json)
