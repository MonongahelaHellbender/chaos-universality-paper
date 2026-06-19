# Phase-Space Volume Contraction, Not the 0-1 Test, Separates Conservative from Dissipative Chaos

**Melissa Ellison — Independent Researcher**

Preprint: see `chaos_universality_classes.pdf`

## What this is

A 26-system computational study showing that the Gottwald-Melbourne 0-1 statistic is
**not** a conservative/dissipative classifier — it is a chaos detector whose finite-time
behavior depends on observable mixing time, not conservation structure. The robust
discriminator is phase-space volume contraction Σλ = ⟨∇·f⟩, which is zero for all
8 Hamiltonian systems tested and strictly negative for all 11 dissipative flows.

The study also shows *why* the 0-1 statistic can mislead: the entire K-vs-length
curve is predicted by the observable autocovariance via a closed-form Green-Kubo
expression (Pearson r=0.955).

## Contents

```
chaos_universality_classes.pdf   — the paper
chaos_universality_classes.tex   — LaTeX source
figures/                         — all 5 figures (PDF)
results/                         — seeded JSON artifacts (reproduce every number in paper)
scripts/                         — figure generation and analysis scripts
```

## Reproduce

```bash
pip install numpy scipy matplotlib scikit-learn
python scripts/chaos_universality_lab.py      # regenerate main results
python scripts/chaos_make_figures.py          # regenerate figures
```

All results are seeded and deterministic. The reported numbers in the paper are read
directly from the JSON files in `results/`.

## Systems

26 systems: 11 dissipative flows, 8 Hamiltonian flows, 3 maps, 1 delay system,
3 non-chaotic controls. Full table in Section 2 of the paper.

## License

MIT — see LICENSE
