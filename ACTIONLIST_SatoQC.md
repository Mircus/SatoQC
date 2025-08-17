# ACTIONLIST — SatoQC (Quasi‑Complex + Hyperfunction Kit)
**Generated:** 2025-08-17 09:18:14


This is a comprehensive, prioritized action plan to take SatoQC from a draft repo to a **runnable, publishable, and test‑covered** toolkit — with a turbulence‑only flow ready for experiments.

---

## P0 — Critical Repo Hygiene (Blockers)

- [ ] **Fix truncated code file**: `src/HyperfunctionCohomology.py` ends mid‑string and contains incomplete blocks. Restore the full source or recompose into modules (see P1).
  - Acceptance: `python -m compileall src` passes; `ruff`/`black` clean; import works.
- [ ] **Add packaging**: create `pyproject.toml` (setuptools or hatch) and `requirements.txt` with explicit deps: `numpy`, `sympy`, `scipy`, `matplotlib` (opt: `jax`, `jaxlib`, `torch`).
  - Acceptance: `pip install -e .` succeeds; `import satoqc` works.
- [ ] **Fix notebooks**: remove hard‑coded absolute paths, ensure relative imports from the package.
  - Acceptance: `examples/quickstart.ipynb` and `examples/satoqc_demo.ipynb` execute end‑to‑end via papermill in CI.
- [ ] **Restore/complete docs**: `README.md`, `ROADMAP.md`, `docs/*.md` contain literal `...` and truncated sections — fill them with full content and working code snippets.
  - Acceptance: All docs render without `...` placeholders; links resolve.

---

## P1 — Module Reorganization (Maintainable API)

Refactor the monolithic `HyperfunctionCohomology.py` into a clear package under `src/satoqc/`:

- [ ] `satoqc/hyperfunction.py` — `Hyperfunction`, boundary values, jump construction.
- [ ] `satoqc/wavefront.py` — Wavefront cones, FBI transform utilities (exact vs approx backends).  
- [ ] `satoqc/atlas.py` — `Chart`, overlap detection, Čech index sets, partition of unity.
- [ ] `satoqc/transitions.py` — `TransitionMap` with **correct Jacobians**, pullback for functions/distributions/hyperfunctions.
- [ ] `satoqc/jtensor.py` — `AlmostComplexStructure` (ACS), `J` parameterizations, local frames/coframes.
- [ ] `satoqc/cohomology.py` — Čech 1‑cocycle assembly, obstruction detection, consistency checks.
- [ ] `satoqc/turbulence.py` — **Energy** \(\mathcal{E}_H\), gradient, and projected flow (turbulence‑only).  
- [ ] `satoqc/diagnostics.py` — Metrics (border index, cocycle violation norms), logging, plots.
- [ ] `satoqc/backends/` — Optional JAX/PyTorch implementations (autodiff, vectorized kernels).

Acceptance: importing each submodule works; `__all__` exposes stable public API.

---

## P2 — Implement Missing Mathematics (Core Correctess)

- [ ] **Transition pullbacks**: Implement general pullback for (hyper)functions via transition maps; carry wavefront cones via pushforward of covectors.  
  - Replace placeholders like `values2_pulled = values2` (currently bypassing pullbacks).
- [ ] **Jacobians**: Implement `TransitionMap.evaluate_jacobian` (currently returns identity). Include symbolic and numeric paths; caching for performance.
- [ ] **FBI transform**: Replace toy `_fbi_transform` with a selectable backend: (i) fast local Gaussian approximations; (ii) numerical quadrature in low‑dim examples; (iii) asymptotic expansions.
- [ ] **Chart overlaps**: Provide robust overlap computation and sampling; expose `compute_chart_overlaps()` in `atlas.py` used by cohomology.
- [ ] **Cech cocycles**: Implement `compute_cech_cocycle()` assembling jump hyperfunctions on overlaps with proper pullbacks.
- [ ] **Wavefront compatibility tests**: Implement cone‑compatibility checks across overlaps and on triple intersections; report `violation_norm` with thresholds.

Acceptance: unit tests in `tests/test_transitions.py`, `tests/test_cech.py`, `tests/test_wavefront.py` pass.

---

## P3 — Turbulence‑Only Energy and Flow (Experiment‑Ready)

- [ ] **Define energy** \(\mathcal{E}_H(J) = \sum_{i<j} \int_{U_{ij}} \|\Delta_{ij}(J)\|^2 w_{ij} \, \mathrm{dvol}\), where \(\Delta_{ij} = \bar\partial_J^{(i)} - \Phi_{ij}^* \, \bar\partial_J^{(j)}\).  
- [ ] **Local \(\bar\partial_J\) stencil** per chart (FD or spectral) for scalars/forms; consistent evaluation of \(J\) at nodes.
- [ ] **Parameterize \(J\)**: either project with \(\Pi_J(A)=\tfrac12(A-JAJ)\) or set \(J=QJ_0Q^{-1}\) with a skew‑param for \(Q\); provide both backends.
- [ ] **Autodiff gradient** for \(\mathcal{E}_H\) (JAX or PyTorch) with projection/retraction per step.
- [ ] **Flow loop**: semi‑implicit stepping, optional DeTurck stabilization, convergence monitors, checkpoints.
- [ ] **Metrics**: border index, # of cocycle violations, energy descent plots, stopping criteria.

Acceptance: `examples/turbulence_flow_T4.ipynb` and `examples/turbulence_flow_S6.ipynb` run and show energy descent; seeds and atlas are deterministic.

---

## P4 — Examples & Datasets

- [ ] **T⁴ example**: 2–4 charts around the torus; standard complex structure plus perturbations; expected \(\mathcal{E}_H \to 0\).
- [ ] **S⁶ example**: 3 charts with the nearly‑Kähler ACS; expected stabilization to nonzero critical energy.
- [ ] **Synthetic atlases**: helpers to generate random atlases/perturbations with fixed seeds (reproducibility).
- [ ] **Visualization**: minimal Matplotlib plots for cones, jumps, and energy curves (no seaborn).

Acceptance: notebooks regenerate figures and match saved reference metrics within tolerance.

---

## P5 — Documentation (User‑Facing & API)

- [ ] **Rewrite README**: Remove ellipses; add install, minimal example, API table, citations, and badges (CI, license, PyPI when ready).
- [ ] **Tutorials** (`docs/`): Complete `SatoQC-Tutorial.md` and `hyperfunctions-tutorial.md` with runnable snippets; avoid pseudo‑code.
- [ ] **API Docs**: Autogenerate via `pdoc` or `mkdocs‑material`; host GitHub Pages.
- [ ] **Design Doc**: Architecture diagram for atlas/transition/J/energy/flow modules; dataflow for \(\Delta_{ij}\) assembly.
- [ ] **Research Notes**: `topos_approach_doc.md` tightened; move speculative parts to `notes/`.

Acceptance: `mkdocs serve` builds; links/code blocks pass doctest where applicable.

---

## P6 — Tests & CI

- [ ] **Unit tests**: transitions, pullbacks, cocycle assembly, energy evaluation, small‑step flow invariants.
- [ ] **Notebook tests**: CI job uses Papermill to execute `examples/*.ipynb` on a small grid.
- [ ] **Static checks**: `ruff`, `black`, `mypy` (where feasible).
- [ ] **GitHub Actions**: `python-package.yml` matrix (3.10–3.12); cache wheels; upload coverage to Codecov.
- [ ] **Pre‑commit**: hooks for formatting/lint; enforce no `...` in docs.

Acceptance: green badges on main; coverage ≥ 80% for core modules.

---

## P7 — Performance & Backends

- [ ] **Vectorization**: replace Python loops with NumPy/JAX vmap where possible.
- [ ] **Caching**: memoize Jacobians/pullbacks on overlaps; reuse stencils between steps.
- [ ] **JAX backend**: optional flag to run energy/gradient with JIT; fall back to NumPy.
- [ ] **Parallel overlap handling**: joblib or Python `concurrent.futures` for large atlases.

Acceptance: reference benchmarks (provided in `benchmarks/`) meet target runtimes on standard laptop.

---

## P8 — Release Engineering

- [ ] **Versioning**: `satoqc.__version__` and CHANGELOG; semantic versioning.
- [ ] **LICENSE & headers**: Ensure HNCL text is included in package data and headers.
- [ ] **PyPI (optional)**: Upload pre‑release `0.1.0a0` once P0–P3 done.
- [ ] **CITATION.cff** with bib entries for Sato and related literature.

---

## Triage: Current Issues Observed

- `src/HyperfunctionCohomology.py` is **truncated/syntax-broken** (file ends mid-string; multiple incomplete blocks).
- **Placeholders** remain (e.g., `TransitionMap.evaluate_jacobian` returns identity; simplified pullbacks; simplified FBI transform).
- **Packaging absent** (`pyproject.toml`, `setup.py`, `requirements.txt` all missing).
- **CI absent** (no GitHub Actions workflows, lint/format/test).
- **Notebooks reference wrong paths** (e.g., `examples/quickstart.ipynb` uses an absolute path under `/mnt/data/extracted_files/...`).
- **Undefined/missing classes or methods** referenced by notebooks (e.g., `QuasiComplexManifold`, `compute_chart_overlaps`, `compute_cech_cocycle`).
- **Docs are truncated** with literal `...` and incomplete sections (e.g., `README.md`, `ROADMAP.md`, `docs/*.md`).
- **No tests** (pytest suite missing).
- **No explicit dependency pins** or optional backends (JAX/PyTorch) declared.
- **No example datasets** or reproducible seeds for charts/atlases beyond minimal stubs.

---

## Milestones & Ownership (suggested)

- **M0 (Week 0–1):** P0–P1 — *Repo hygiene & reorg* (Owner: Core maintainer).  
- **M1 (Week 2):** P2 — *Math completeness & tests* (Owner: Math/Algo).  
- **M2 (Week 3):** P3 — *Turbulence flow prototype* (Owner: Algo/Backend).  
- **M3 (Week 4):** P4–P5 — *Examples & docs* (Owner: Docs).  
- **M4 (Week 5):** P6–P7 — *CI & perf* (Owner: Infra).  
- **M5 (Week 6):** P8 — *Pre‑release 0.1.0a0* (Owner: Maintainer).

---

## File‑Level Fix List (initial)

- [ ] `src/HyperfunctionCohomology.py`: restore completeness; split into `src/satoqc/*.py` modules (see P1); remove placeholder returns (`np.eye`) and finalize FBI/pullback logic.
- [ ] `examples/quickstart.ipynb`: replace absolute module path with `import satoqc as sqc`; update cells to match new API.
- [ ] `examples/satoqc_demo.ipynb`: same; ensure it runs headless in CI.
- [ ] `README.md`/`ROADMAP.md`/`docs/*.md`: remove `...`; fill sections and working examples; verify markdown links.
- [ ] Add `pyproject.toml`, `requirements.txt`, `.github/workflows/python-package.yml`, `pytest.ini`, `.pre-commit-config.yaml`.

---

### Notes

- Keep the **topos** content in a clearly labeled research note; main kit remains “topos‑free” to stay runnable.
- Prefer **deterministic** examples (seeded atlas/J generation).
