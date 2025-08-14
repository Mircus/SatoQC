# ROADMAP — Quasi‑Complex + Hyperfunction Kit (Topos‑free)

This roadmap turns the current repo into a **runnable kit** for analyzing quasi‑complex structures via **hyperfunctions**, **microlocal compatibility**, and **Čech gluing** — with **no topos machinery**. It is organized by milestones, concrete tasks, acceptance criteria, and file structure.

---

## Guiding Principles (Scope for this phase)
- Work at the **chart/atlas + hyperfunction** level only.
- Provide **concrete, reproducible computations**: jumps across overlaps, WF transforms, curved Dolbeault diagnostics, simple indices.
- Ship minimal, clean **Python package** + **examples** + **tests**.
- Defer: classifying topoi, non‑abelian cohomology, Spencer formalism (future phase).

---

## Milestone M0 — Package Bootstrap (Day 0–1)
**Goal:** Make the repo installable and testable; fix current breakages.

**Tasks**
- [ ] Create `pyproject.toml` with package name `satoqc` and deps: `numpy`, `sympy`, `scipy`, `matplotlib`, `networkx`, `pytest`.
- [ ] Move code into `satoqc/` package; remove `...` placeholders; ensure module imports don’t break.
- [ ] Add `requirements.txt` (mirror of deps), `LICENSE`, `.gitignore`.
- [ ] Add `tests/` with a dummy test to validate CI setup.
- [ ] Pin minimal Python version (3.10+).

**Acceptance Criteria**
- `pip install -e .` works locally.
- `pytest` runs and passes the dummy test.
- `examples/quickstart.ipynb` imports `satoqc` without errors (even if demos are stubs).

---

## Milestone M1 — Core Data Model (Day 1–3)
**Goal:** Minimal objects to express charts, transitions, J, hyperfunction boundary values, and overlaps.

**Deliverables**
- `satoqc/atlas.py`
  - `Chart(name, domain: Callable[np.ndarray->bool], phi_to: Dict[str, Callable])`
  - `Atlas(charts: Dict[str,Chart]).overlaps() -> List[Tuple[str,str]]` (sampling or metadata)
- `satoqc/jstructure.py`
  - `JStructure(J: Callable[x->np.ndarray])`
  - `verify(x, tol)`; `project_01(vec, x)`; `dbar(grad_f, x)`; `curvature02_proxy(f, x)`
- `satoqc/hyperfun.py`
  - `Hyperfunction(chart, boundary_expr: Callable, wf_cone: np.ndarray)`
  - `pushforward(phi, dphi)`; `pullback(phi, dphi)`

**Acceptance Criteria**
- Unit tests for basic `J^2 = -I` verification on sample points.
- WF cone pushforward/pullback tested on linear maps.

---

## Milestone M2 — 45° Rotation Demo (Day 3–5)
**Goal:** End‑to‑end computation on the canonical two‑chart example.

**Deliverables**
- `satoqc/examples_45deg.py`
  - Define two charts `U1, U2` on `R^4`, with 45° rotation in (u,v) on `U2`.
  - Provide local holomorphic boundary values `[F1], [F2]` and compute the **jump**:
    \[
      c_{12} = [F_1] - [F_2] 
      = \delta_{\text{border}}\Big[(1-1/\sqrt{2})(u+v) + i\,\big(v-(v-u)/\sqrt{2}\big)\Big].
    \]
  - Implement WF rotation: `(xi_u, xi_v) -> ((xi_u - xi_v)/√2, (xi_u + xi_v)/√2)`.
  - Compute a **border index** `Ind_border(J)` by pairing WF with the conormal to the overlap strip.
- CLI: `python -m satoqc.examples_45deg` prints:
  - Non‑integrable verdict (nonzero jump class).
  - Border index value.
  - Sanity check on WF rotation.

**Tests**
- `tests/test_45deg_jump.py`: assert jump != 0 on sampled overlap; assert WF transform matches 45° formula.

**Acceptance Criteria**
- Running the demo prints the expected symbolic jump and a nonzero index.
- Tests green.

---

## Milestone M3 — Curved Dolbeault Diagnostics (Day 5–7)
**Goal:** Provide the user with a practical way to **measure nonintegrability** chartwise.

**Deliverables**
- `satoqc/jstructure.py`:
  - `dbar_J(f, x)` for scalar fields (numerical gradient + (0,1) projection).
  - `curvature02_proxy(f, x, eps)` approximating \(\bar\partial_J^2 f\) via finite differences (reports a (0,2) “curvature mass”).

- `satoqc/cech.py`:
  - `jump(h_left, h_right, phi_right_to_left)` → hyperfunction difference on overlap.
  - `cech_1_cocycle(jumps) -> dict` and `nonvanishing(jumps) -> bool`.

**Tests**
- `tests/test_dbar_curvature_proxy.py`: curvature proxy nonzero on 45° example overlap.
- `tests/test_cech_basic.py`: synthetic cocycle is nonvanishing when one jump is nonzero.

**Acceptance Criteria**
- Diagnostic prints a nonzero curvature proxy on the 45° example.
- Čech utilities work on a toy three‑chart cover (unit test).

---

## Milestone M4 — S^6 Minimal Module (Day 7–10)
**Goal:** Numerically demonstrate nonintegrability for the \(G_2\) almost complex structure, no topos.

**Deliverables**
- `satoqc/s6.py`
  - `cross_product(x, y)` in \(\mathbb{R}^7\), `J_x(v) = x × v` restricted to `T_x S^6`.
  - Random orthonormal tangent frames; intrinsic check: \(N_J(u,v) \approx -2 (u×v)^\top\) nonzero on average.
  - Two‑chart (north/south) cover with a **proxy equator jump index** (reusing border index machinery).

**Tests**
- `tests/test_s6_nijenhuis.py`: sample points/frames → mean \(\|N_J\|\) > threshold.
- `tests/test_s6_equator_index.py`: index nonzero on equator samples.

**Acceptance Criteria**
- Running `python -m satoqc.s6` prints nonzero Nijenhuis diagnostic and equator index.

---

## Milestone M5 — Docs, Notebook, and Stability (Day 10–12)
**Goal:** Ship a coherent kit for Struppa’s group.

**Deliverables**
- Update `README.md` with 3‑minute quickstart:
  1. `pip install -e .`
  2. `python -m satoqc.examples_45deg`
  3. `python -m satoqc.s6`
- Rewrite `examples/quickstart.ipynb` to call the new APIs; include plots of WF cones and border strip.
- Add `CHANGELOG.md`, `CONTRIBUTING.md`, and simple `Makefile` targets: `format`, `test`, `lint`, `demo`.
- CI: GitHub Actions for `pytest` on 3.10–3.12; Ruff/Black formatting.

**Acceptance Criteria**
- Notebook runs top‑to‑bottom without edits.
- CI green on main.

---

## Nice‑to‑Have (post‑M5, optional)
- Parametrix `T_J` (Bochner–Martinelli style) in `satoqc/kernels.py`; empirical check that `dbar_J T_J ≈ I` up to curvature error.
- Visualization helpers for conormal pairing and WF rotation fields.
- Small dataset of synthetic atlases for benchmarking (JSON configs).

---

## Anti‑Goals (explicitly out of scope now)
- No classifying topoi, Spencer cohomology, or non‑abelian H^1.
- No internal spectral sequences.
- No heavy symbolic Nijenhuis machinery beyond the S^6 numeric diagnostic.

---

## Roles / Ownership (suggested)
- **Core API & Examples:** one lead (kit owner).
- **Numerics & Tests:** one engineer to harden finite‑diff and random frame sampling, write tests.
- **Docs & Notebook:** one author to keep README + notebook aligned.

---

## File/Module Plan (final state for this phase)

```
satoqc/
  __init__.py
  atlas.py                # Chart, Atlas, overlaps
  jstructure.py           # JStructure, dbar_J, curvature02_proxy
  hyperfun.py             # Hyperfunction proxy, WF pushforward/pullback
  cech.py                 # jump, 1-cocycle, nonvanishing
  microlocal.py           # WF transforms, conormal pairing, border index
  examples_45deg.py       # runnable demo (CLI)
  s6.py                   # S^6: J via cross-product, Nijenhuis numeric, equator index
  viz.py                  # optional plotting
examples/
  quickstart.ipynb
tests/
  test_45deg_jump.py
  test_dbar_curvature_proxy.py
  test_cech_basic.py
  test_s6_nijenhuis.py
  test_s6_equator_index.py
pyproject.toml
README.md
CHANGELOG.md
CONTRIBUTING.md
LICENSE
```

---

## Success Criteria (what “done” looks like)
- Running `examples_45deg` prints the symbolic jump and a **nonzero border index**.
- Running `s6` prints a **nonzero Nijenhuis diagnostic** and an **equator index**.
- `pytest` ≥ 90% pass rate (100% for this phase).
- Quickstart notebook executes cleanly and matches README.
