# Neural Dynamics of Object Recognition Under Sensory Uncertainty

*Research Project for the Neuromatch Academy's Computational Neuroscience Course.*

This repository contains the complete Python pipeline for analyzing intracranial electrocorticography (ECoG) data to investigate how the human brain represents visual objects under varying levels of sensory noise. The project uses a combination of traditional signal analysis (High-Gamma Activity, ERPs) and machine learning (Multivariate Pattern Analysis - MVPA) to characterize the spatio-temporal dynamics of face and house perception.

Note: This repository was adapted and expanded from the work of my project peer Mohammadreza Shahsavari (https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty). Since we worked as a team, this repository represents the contributions of all team members, not just me.
The latest commits (September2026) are the final phase of this repository that include statistical test to produce P-values for our hypotheses.

## 🎯 Project Goal

The primary goal of this research is to map the evolution of categorical information in the brain and understand how these neural codes are systematically altered by sensory uncertainty. We analyze ECoG data from subjects performing a visual task where they identify images of faces and houses obscured by different levels of noise.

---

## 📈 Key Findings

Our analysis reveals a robust yet nuanced process of object recognition in the brain. While neural representations are highly specialized, they are impacted by noise in distinct ways.

### 1. Selective High-Gamma Activity (HGA)

Initial qualitative analysis of high-gamma activity (HGA) shows a strong, localized neural response to specific visual categories. As hypothesized, certain cortical areas are highly selective, with some channels showing a dramatic increase in activity specifically for face stimuli, and others for houses.

*The figure below shows the average broadband response across 50 ECoG channels for one subject. Note the clear selectivity: channel **ECOG_047** responds strongly to faces (orange line), while channel **ECOG_031** responds to houses (blue line).*

<p align="center">
  <img src="High%20Gamma%20Activity%20%26%20ERP%20Plots/SubjectIndx1%20-%20NoiseLevel%20All.png" width="800">
</p>

### 2. The "Breaking Point" in Neural Decoding

Using a sliding-window MVPA approach with an SVM classifier, we decoded the object category (face vs. house) from the neural data at different noise levels. Our results show that the brain's ability to represent these categories is remarkably resilient up to a certain point, after which it degrades abruptly.

- **Decreasing Accuracy with Noise**: As sensory noise increases, the peak decoding accuracy systematically decreases. The neural representation is strongest in low-noise conditions and weakens as the stimulus becomes more ambiguous.

- **The 40-50% Noise "Breaking Point"**: Neural representation does not degrade linearly. It drops sharply across the 40–50% noise range rather than gradually, suggesting a "breaking point" in the evidence accumulation process. Formal group-level statistics (see [Statistical Results](#-statistical-results-statipynb)) confirm this: accuracy stays significantly above chance in every tested bin, but the drop from the 0.2–0.4 to the 0.4–0.6 bin is the only adjacent-bin decrease that survives FDR correction, and a segmented regression locates a slope change at ≈ 0.53 noise.

*Left: Decoding accuracy curves for different noise bins. Note that the purple (0-20% noise) and blue (20-40% noise) curves show the strongest decoding, which weakens at higher noise levels. Right: Peak decoding accuracy drops as noise increases.*

<p align="center">
  <img src="Across%20Noise%20Levels%20Across%20Subjects%20Results/noise_levels_comparison.png" width="500">
  <img src="Across%20Noise%20Levels%20Across%20Subjects%20Results/PeakDecodingAccuracy%20vs%20NoiseLevel.png" width="300">
</p>

This abrupt cutoff is also visible in the channel-wise MVPA results for a single subject, where the robust decoding seen in low-noise bins (1 and 2) vanishes in higher-noise bins.

<p align="center">
  <img src="Across%20Noise%20Levels%20Across%20Subjects%20Results/photo_2025-08-03_06-38-05.jpg" width="800">
</p>

### 3. Time Delay in Neural Processing

Increased sensory noise also introduces a delay in the peak neural representation. The grand-average heatmap across all subjects shows that as noise increases, the "hotspot" of maximum decoding accuracy not only gets weaker but also shifts rightward in time. This indicates that the brain requires more time to accumulate sufficient evidence to represent the object when the signal is noisy.

*Decoding accuracy heatmap averaged across subjects. The peak accuracy (yellow/orange) occurs later for the 0.2-0.4 noise level compared to the 0.0-0.2 level, demonstrating a temporal delay in processing.*

> **Caveat.** The group-level test does **not** support this visual impression:
> peak latency did not increase reliably with noise (see
> [Statistical Results](#-statistical-results-statipynb)), and much of the
> apparent rightward shift in the heatmap is not statistically robust.

<p align="center">
  <img src="Across%20Noise%20Levels%20Across%20Subjects%20Results/noise_levels_heatmap.png" width="700">
</p>

---

## 📊 Statistical Results (stat.ipynb)

Group-level statistics for all four hypotheses are computed in `stat.ipynb`
(reusable functions in `statistical_tests.py`). All tests are built on
**statsmodels**: OLS-based one-sample t-tests, linear mixed-effects models with
a random intercept per subject, Benjamini-Hochberg FDR correction, and
segmented (piecewise) regression. The input is the real `main.py` output:
**7 subjects × 5 noise bins × 10 MVPA repetitions** (35 subject-level and 350
repetition-level observations). Full results are saved to
`statistics_summary.csv`.

Every mixed-effects slope is cross-checked with a **subject-level cluster-robust
OLS**, and the confirmatory H1/H3 tests run on **one observation per subject per
noise bin** (n = 7), so no subject is pseudo-replicated.

| Hypothesis | Test | Statistic | p-value | Supported? |
|---|---|---|---|---|
| **H1** — decoding accuracy above chance (0.5) | One-sample t-test (OLS intercept) on peak accuracy, per noise bin | t(6) = 8.60 (lowest-noise bin) | 6.8e-05 | ✅ Yes — above chance in **every** noise bin after FDR correction (largest corrected p = 6.8e-05); pooled over bins t(6) = 18.58, p = 7.8e-07 |
| **H2** — accuracy decreases with sensory noise | Linear mixed-effects model (random intercept per subject) | slope = −0.184 acc./noise | 1.1e-05 (cluster-robust OLS; mixed model 2.7e-67) | ✅ Yes |
| **H3** — "breaking point" near 40–50% noise | (a) Per-bin tests vs chance, FDR-corrected — all bins stay above chance, so there is no bin-level breaking point; (b) paired t-test on adjacent bins, FDR-corrected — significant drop at the 0.2–0.4 → 0.4–0.6 boundary and only there; (c) segmented regression — breakpoint ≈ 0.53 with significant slope change | t(6) = 4.76 (adjacent-bin cliff); hinge slope change = 0.285 | 0.013; 2.6e-04 | ✅ Yes — steep drop followed by a plateau |
| **H4** — peak decoding latency increases with noise | Linear mixed-effects model on peak latency | slope = −331 ms/noise | 1.0 (one-sided; cluster-robust OLS agrees) | ❌ Not supported on the current real run — latency *decreased* with noise |

**Interpretation.** The real decoder output robustly confirms H1 and H2. For
H3, accuracy does **not** return to chance in any tested bin; instead it drops
sharply across the 0.2–0.4 → 0.4–0.6 boundary (the only adjacent-bin drop that
survives FDR correction) and then plateaus at ≈ 0.58 — the 40–50% "cliff"
predicted by the hypothesis, corroborated by the segmented-regression
breakpoint at ≈ 0.53 (piecewise AIC −1034.6 vs linear AIC −993.2).
H4 is not supported: in this run peak latency shortened with noise, so the
one-sided test correctly fails to reject in the hypothesized direction.

### Robustness notes

- **Peak-search window.** `peak_accuracy` / `peak_time` are the argmax of the
  accuracy curve over the whole `−200…500 ms` window, so in high-noise bins a
  noisy *pre-stimulus* window can be selected as the "peak": 4/35 subject-level
  and 79/350 repetition-level peaks fall at or before 0 ms, almost all of them
  in the 0.4–0.8 noise bins. This is the main reason H4 comes out negative.
  Restricting the peak search to post-stimulus windows
  (`load_group_mvpa_results(BASE_DIR, min_peak_time=0.0)`) removes every
  baseline peak and moves the H4 slope from −331 to −206 ms/noise — the
  hypothesis is still not supported. The H1/H3 conclusions are unaffected.
- **Mixed-model convergence.** The between-subject variance in H2 is estimated
  at ≈ 0, so that mixed model collapses onto OLS, and the H4 mixed model does
  not reach convergence (`Converged: No`). Both slopes are therefore reported
  together with their cluster-robust OLS counterparts, which are the values to
  trust (H2 p = 1.1e-05, H4 p = 1.0).
- **Non-independence of repetitions.** The 10 repetitions are random train/test
  splits of the *same* trials, so repetition-level rows are not fully
  independent and repetition-level p-values are optimistic. All confirmatory
  tests (H1, H3) run on subject-level data; the repetition-level models (H2, H4)
  are always paired with subject-clustered standard errors.
- **Breakpoint selection.** The segmented-regression breakpoint is chosen by
  AIC over a grid of candidates, so its hinge p-value is descriptive rather than
  confirmatory — it corroborates the adjacent-bin paired t-test rather than
  standing alone.

---

## 📂 Repository Structure

This repository is organized into several key scripts:

| File                               | Description                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                          | **Main executable script.** Configures and runs the entire MVPA pipeline across all specified noise ranges and subjects.              |
| `utils.py`                         | Core utility functions, including data loading, preprocessing, and the `MVPAAnalyzer` class which implements the decoding analysis. |
| `stat.ipynb`                       | Jupyter notebook implementing all group-level hypothesis tests (H1–H4) on the MVPA results; produces `statistics_summary.csv`.     |
| `statistical_tests.py`             | Reusable statsmodels-based functions for the H1–H4 hypothesis tests: result loading (`load_group_mvpa_results`, with an optional `min_peak_time` post-stimulus peak window) plus a labelled simulated-data fallback (`simulate_group_dataset`). Imported by `stat.ipynb`. |
| `statistics_summary.csv`           | Machine-readable summary of every hypothesis test, statistic and p-value from the latest run.                                      |
| `generate_visualizations.py`       | A standalone script to generate summary plots and visualizations (Levels 1, 2, and 3) from the saved results of the `main.py` pipeline. |
| `Highy Gamma Activity & ERP Analysis.py` | Script for exploratory data analysis, including plotting HGA and ERPs for specific channels and noise levels.               |
| `Loading & Preprocessing Functions.py` | Contains helper functions for loading and preprocessing the ECoG data, designed to be make preprocessing pipeline easier to understand.              |
| `_validate_loader.py`              | Quick sanity-check script for the MVPA-results loader used by the statistical pipeline.                                           |
| `pyproject.toml` / `uv.lock`       | Project metadata and locked dependencies (managed with [uv](https://docs.astral.sh/uv/)).                                          |
| `requirements.txt`                 | A list of all the necessary Python packages to run the code in this repository.                                                           |

---

## 🚀 How to Use

Follow these steps to replicate the analysis.

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Data Source

The raw ECoG data (`faceshouses.npz`) used in this project can be downloaded from the Stanford Digital Repository:
- **Link**: [https://exhibits.stanford.edu/data/catalog/zk881ps0522](https://exhibits.stanford.edu/data/catalog/zk881ps0522)

### 3. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/merhan-jaberi/ECoG-Object-Recognition-Uncertainty.git
cd ECoG-Object-Recognition-Uncertainty
```
> Note: the repository is **private** — authenticate with your GitHub account
> when cloning or pulling.

Next, install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Running the MVPA Pipeline

The entire MVPA is controlled by the `main.py` script.

1.  **Update the Data Path**: Open `main.py` and **you must update the file path** in the `config` dictionary to point to the location where you saved `faceshouses.npz`.

    ```python
    # In main.py
    config = {
        # ...
        'filepath': r'C:\path\to\your\data\faceshouses.npz',
        # ...
    }
    ```

2.  **Configure the Analysis**: You can modify the rest of the `config` dictionary to set your desired parameters. This includes preprocessing steps, MVPA windowing parameters, or processing mode (`ecog` vs. `high_gamma`).

3.  **Define Noise Bins**: Modify the `noise_ranges_to_analyze` list to specify which noise intervals you want to analyze. The script will run a separate, complete analysis for each entry.

    ```python
    # In main.py
    noise_ranges_to_analyze = [
        [0.0, 0.2],
        [0.2, 0.4],
        # ... and so on
    ]
    ```

4.  **Execute the Script**: Run the analysis from your terminal.

    ```bash
    python main.py
    ```

The script will create output directories based on the configuration for each noise range (e.g., `mvpa_results - ecog - ... - noise0.00to0.20`). If results for a subject already exist, that subject will be skipped to allow for easy resumption of a long analysis.

> **Version control note:** the raw data file (`faceshouses.npz`, ~620 MB) and the
> intermediate `mvpa_results ...` folders (~530 MB) are excluded from this
> repository (see `.gitignore`) — the latter are regenerated by running
> `main.py`.

### 5. Generating Summary Visualizations

After the MVPA pipeline has finished and the result files (`.pkl`) are generated, you can create the summary figures.

1.  **Configure the Script**: Open `generate_visualizations.py` and set the `base_results_dir` to the parent directory containing all your analysis folders. In most cases, this will just be the project's root directory.

    ```python
    # In generate_visualizations.py
    base_results_dir = r"."
    ```

2.  **Execute the Script**: Run the script from your terminal.

    ```bash
    python generate_visualizations.py
    ```

This will create a new folder named `NoiseRange_Visualizations` containing three levels of summary plots, including the heatmaps and comparison plots shown in the findings section.

### 6. Running the Statistical Tests

After the MVPA pipeline has produced the `mvpa_results ...` folders, open
`stat.ipynb` in Jupyter and run all cells. It automatically loads the real
decoding results and executes the four hypothesis tests (H1–H4), saving the
summary table to `statistics_summary.csv`.

```bash
jupyter notebook stat.ipynb
```

The analysis settings live at the bottom of the setup cell:

```python
BASE_DIR   = "."          # folder holding the 'mvpa_results - ...' output of main.py
CHANCE     = 0.5          # chance decoding accuracy for 2 classes
ALPHA      = 0.05         # significance level
CORRECTION = "fdr_bh"     # multiple-comparison correction for H3 ('bonferroni' also allowed)
```

Use the project's `.venv` interpreter as the notebook kernel. If no
`mvpa_results ...` folders are present, the notebook prints a warning and falls
back to a clearly-labelled **simulated** dataset (`st.simulate_group_dataset`)
so the pipeline can still be demonstrated — re-run `main.py` and point
`BASE_DIR` at its output to analyse real data. For single-channel decoding
instead of the all-channel decoder, use
`st.load_group_mvpa_results(BASE_DIR, source="single_channels")`.

---
## 🛠️ Dependencies

The main libraries used in this project are:
- `numpy`
- `mne`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `nilearn`
- `plotly`


