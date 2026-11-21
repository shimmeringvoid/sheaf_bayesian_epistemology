# Truth as a Structural Invariant: A Sheaf-Bayesian Epistemology for Superintelligence

**Author:** Rafael Espericueta ([ShimmeringVoid.org](https://ShimmeringVoid.org))  
**Manuscript (v8.7):** [`sheaf_bayesian_epistemology_v8_7.pdf`](sheaf_bayesian_epistemology_v8_7.pdf) | **LaTeX Source:** [`sheaf_bayesian_epistemology_v8_7.tex`](sheaf_bayesian_epistemology_v8_7.tex) | **Verification Suite:** [`simulate_sheaf_bayes_v8_6.py`](simulate_sheaf_bayes_v8_6.py)

> **Start Here — Plain-Language Guide:** Read [**`OVERVIEW.md`**](OVERVIEW.md) for an accessible, non-technical tour of the core ideas, the three-layer architecture, and what the four Python experiments test.

---

## Overview

Next-token maximum likelihood estimation (MLE) gives large language models remarkable linguistic fluency, yet likelihood under a pooled corpus is neither necessary nor sufficient for truth. Building on Lopez-Paz's *Against All Likelihood* (2026), this repository contains the manuscript, macro-linked LaTeX source, and reproducible Python verification suite developing a three-layer Bayesian and cellular-sheaf epistemology:

1. **Layer 1 — Hierarchical Testimony, Provenance Clustering, and Capped Calibration:** Shows that pooled MLE weights latent sources by corpus prevalence rather than reliability; proves single-channel testimony saturation ($A/C$), exponential independent corroboration ($\alpha_1^K$), sub-exponential unverified scaling under common-origin floor and Chinese Restaurant Process priors, and verifier-anchored resilience against paraphrase attacks, passing exact expected inverse block sizes $\hat{s}_v = \mathbb{E}_{\Pi \mid W}[1/\vert{}b(v)\vert{}]$ and capped probit precisions downstream.
2. **Layer 2 — Mechanistic Consilience and Exploratory GFlowNets:** Separates nuisance framing symmetries from substantive epistemic angles, pairing off-policy exploratory Trajectory Balance GFlowNets with a scope-gated, verified-signature Product-of-Experts consilience reward and Chow's risk-optimal abstention rule.
3. **Layer 3 — Harmonically Centred, Exponent-Weighted Knowledge Sheaves:** Models cross-domain integration on cellular sheaves over a regular two-complex, proves Weyl–Wedin stability of approximate harmonic subspaces $H^0_{\tau_{\mathrm{harm}}}$, derives an exact quadratic EM odds-contraction map with breakdown threshold $S_{\mathrm{fault}} < S_{\mathrm{hon}}$, and evaluates two-stage (weighted spatial-median centring + exponent-weighted Student-$t$ EM) and three-stage (map-aware coupling + fit initialization) sheaf filters alongside dial-free $2 \times 2$ spacetime sheaf selection and a two-step Hodge curl/$H^1$ fault rule.

Every experimental table body and inline empirical scalar in Section 6 of the paper is exported automatically as a `\res...` macro into [`v8_6_results_macros.tex`](v8_6_results_macros.tex) by [`simulate_sheaf_bayes_v8_6.py`](simulate_sheaf_bayes_v8_6.py) (`seed=42`), with zero hand-typed fallback numbers in the LaTeX manuscript. (Version 8.7 updates only the collaboration statement relative to Version 8.6; its script, exported macros, figures, and arXiv abstract are those of Version 8.6.)

---

## Repository Contents

| File | Description |
| :--- | :--- |
| [`OVERVIEW.md`](OVERVIEW.md) | Accessible, plain-language guide to the paper and Python experiments |
| `sheaf_bayesian_epistemology_v8_7.pdf` | Compiled 32-page manuscript (Version 8.7) |
| `sheaf_bayesian_epistemology_v8_7.tex` | Main LaTeX source (inputs `v8_6_results_macros.tex` with hard check) |
| `simulate_sheaf_bayes_v8_6.py` | Self-contained simulation, audit, macro exporter, and plotting script |
| `v8_6_results_macros.tex` | Auto-generated LaTeX table rows and inline empirical macros (438 macros) |
| `fig1_testimony_saturation_v8_6.png` | Experiment 1: Utterance saturation & subset-DP provenance / forgery |
| `fig2_sheaf_robustness_v8_6.png` | Experiment 2: $SO(2)$ near-origin breakdown & distance error sweep |
| `fig3_spacetime_sheaf_v8_6.png` | Experiment 3: Dial-free 2D Lorentzian vs. Galilean spacetime sheaves |
| `v8_6_arxiv_abstract.txt` | Plain-text / math abstract formatted for arXiv metadata (1,918 chars) |
| `environment.yml` / `requirements.txt` | Conda and `pip` environment specifications |

---

## Reproducing All Results and Compiling the Paper

### 1. Python Environment

**Option A — Conda (recommended):**
~~~bash
conda env create -f environment.yml
conda activate sheaf
~~~

**Option B — Standard `venv` + `pip`:**
~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

### 2. Run the Verification Suite & Byte-Check Macros

~~~bash
cp v8_6_results_macros.tex v8_6_results_macros.ref
python3 simulate_sheaf_bayes_v8_6.py
cmp -s v8_6_results_macros.ref v8_6_results_macros.tex && echo "MACRO CHECK PASSED: BYTE-IDENTICAL"
rm v8_6_results_macros.ref
~~~

Pass `--no-plots` to skip `matplotlib` figure rendering and regenerate `v8_6_results_macros.tex` only:

~~~bash
python3 simulate_sheaf_bayes_v8_6.py --no-plots
~~~

### 3. Compile the LaTeX Manuscript

~~~bash
pdflatex -interaction=nonstopmode -halt-on-error sheaf_bayesian_epistemology_v8_7.tex
pdflatex -interaction=nonstopmode -halt-on-error sheaf_bayesian_epistemology_v8_7.tex
~~~

---

## Citation

~~~bibtex
@misc{espericueta2026sheafbayes,
  title  = {Truth as a Structural Invariant: A Sheaf-Bayesian Epistemology for Superintelligence},
  author = {Espericueta, Rafael},
  year   = {2026},
  month  = nov,
  note   = {Preprint and reproducible verification suite, ShimmeringVoid.org},
  url    = {https://github.com/shimmeringvoid/sheaf_bayesian_epistemology}
}
~~~

## License

* **Code (`*.py`):** [MIT License](LICENSE)
* **Manuscript & Figures (`*.tex`, `*.pdf`, `*.png`, `*.md`):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
