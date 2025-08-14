# Applications — Quasi‑Complex + Hyperfunction Kit (Topos‑free)

This document lists **practical use cases** you can tackle immediately with the kit, focusing on
chart/atlas data, hyperfunction boundary values, microlocal (WF) compatibility, and Čech gluing.
No topos machinery is required.

---

## Contents
1. [Complex Geometry Diagnostics](#complex-geometry-diagnostics)
2. [Numerical PDE & Simulation Models](#numerical-pde--simulation-models)
3. [Microlocal Defect Detection (Imaging/Inversion)](#microlocal-defect-detection-imaginginversion)
4. [Computational Algebraic Geometry](#computational-algebraic-geometry)
5. [Geometry‑Aware Data Science](#geometry-aware-data-science)
6. [Education & Research Prototyping](#education--research-prototyping)
7. [KPIs / Outputs the Kit Provides](#kpis--outputs-the-kit-provides)
8. [Integration Checklist](#integration-checklist)

---

## Complex Geometry Diagnostics

**Who:** Applied geometers, mathematical physicists.  
**Problem:** Test whether a numerically‑defined almost complex structure \(J\) is near‑integrable, and identify where it fails.  
**How:**

- Build an atlas of charts and transitions.
- Attach local boundary values \([F_\alpha]\) (polynomial/symbolic proxies or sampled fields).
- Compute **jumps** on overlaps: \(c_{\alpha\beta}=[F_\alpha]-\phi_{\alpha\beta}([F_\beta])\).
- Evaluate **WF transforms** under transitions to check microlocal compatibility.
- Use **curved Dolbeault** diagnostics (\(\bar\partial_J\), curvature (0,2) proxy).

**Micro‑example (pseudo‑code):**
```python
from satoqc import atlas, jstructure, hyperfun, cech, microlocal

J = jstructure.JStructure(J_callable)         # J(x) matrix
U1, U2 = atlas.Chart(...), atlas.Chart(...)
A = atlas.Atlas({"U1": U1, "U2": U2})

F1 = hyperfun.Hyperfunction("U1", boundary_expr=f1, wf_cone=wf1)
F2 = hyperfun.Hyperfunction("U2", boundary_expr=f2, wf_cone=wf2)

c12 = cech.jump(F1, F2, phi_right_to_left=U2.phi_to["U1"])
idx = microlocal.border_index(c12, overlap=U1∩U2)

print("Non‑integrable" if cech.nonvanishing([c12]) else "Integrable?")
print("Border index:", idx)
```

---

## Numerical PDE & Simulation Models

**Who:** CFD, plasma physics, general relativity numerics.  
**Problem:** Coordinate ansätze presume complex structure (for symmetry reduction). Detect **early** if the assumption is inconsistent across chart overlaps.  
**How:**
- Extract \(J\) and fields from numerical grids per chart.
- Fit simple local analytic boundary values (least squares, splines) and compute **jumps**.
- Track WF cones of residuals to see if mismatches are geometric or numerical.

**Outcome:** Flag **bad coordinates** before burning compute time; suggest where to refine mesh or alter chart layout.

---

## Microlocal Defect Detection (Imaging/Inversion)

**Who:** Seismic/SAR/CT inversion.  
**Problem:** Reconstructed fields are glued from local models; inconsistencies concentrate along interfaces.  
**How:**
- Treat local reconstructions as \([F_\alpha]\); compute **jump hyperfunctions** along acquisition seams.
- Pair the jump’s **WF** with the **conormal** to the seam to get a **defect index**.
- Map indices to 2D/3D heatmaps to reveal acquisition/model inconsistencies.

**Outcome:** Localize misfit seams, differentiate model error vs. acquisition artifacts.

---

## Computational Algebraic Geometry

**Who:** Numerical AG / complex analysts.  
**Problem:** Decide if a patchwise parametrization is globally complex‑analytic.  
**How:**
- Use polynomial/rational boundary proxies on charts; compute Čech **1‑cocycle** of jumps.
- If non‑vanishing, stop early; if small, try local **twists** (gauge) to reduce curvature and test again.

**Outcome:** Early failure certificate (non‑integrability) without heavy Gröbner machinery.

---

## Geometry‑Aware Data Science

**Who:** Manifold learning, diffusion MRI, shape analysis.  
**Problem:** Pipelines assume data lie on complex manifolds; check and score that assumption.  
**How:**
- Fit a quasi‑complex atlas to embeddings; compute **non‑integrability scores** (border index, curvature proxy).
- Visualize **where** the data manifold violates complex compatibility.

**Outcome:** Select models/embeddings that respect true geometry; improve generalization.

---

## Education & Research Prototyping

Run the kit on:
- **45° rotation demo** (two charts on \(\mathbb{R}^4\)): explicit jump and WF rotation known in closed form.
- **\(S^6\)** minimal demo: numeric Nijenhuis diagnostic + equator “border index”.

Use these as templates for your own manifolds/atlases.

---

## KPIs / Outputs the Kit Provides

- **Jump hyperfunctions** \(c_{\alpha\beta}\) on overlaps (symbolic or sampled).  
- **WF transforms** under transitions; compatibility verdicts.  
- **Border/Equator index**: scalar microlocal defect per seam/overlap.  
- **Curved Dolbeault diagnostics**: \(\bar\partial_J\) residuals and (0,2) curvature proxy.  
- **Čech 1‑cocycle status**: quick non‑vanishing checks.

These serve as decision criteria for “integrability?”, “where?”, and “how severe?”.

---

## Integration Checklist

- [ ] Provide chart domains and transition maps (or grid‑sampled approximations).  
- [ ] Supply a callable \(J(x)\) or a per‑cell matrix field.  
- [ ] Choose local boundary proxies \([F_\alpha]\) (polys/splines/NNs).  
- [ ] Define a seam/overlap sampling strategy.  
- [ ] Run: jump → WF compatibility → indices → curved Dolbeault diagnostics.  
- [ ] Thresholds: set acceptable bounds for indices/residuals for your application.

---

## Notes

- Everything here is **topos‑free**: it uses hyperfunction boundary values as a **computational model** and standard microlocal operations (WF pushforward/pullback).
- The same code scales from didactic 2D/4D setups to multi‑chart, high‑dimensional numerical models.
