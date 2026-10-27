# DAE System Model Calibration and Validation

Python toolkit to **calibrate (PSO), validate and diagnose (FIM, t-values, correlations, information criteria)** CasADi-based DAE bioprocess models against time-series data. It was developed for *Mycobacterium smegmatis* batch cultivation on glycerol and ammonium.

Two models are included:

- **Baseline** (Monod), from Díaz-Figueroa et al. 2025.
- **Extended** (Droop quota), adding an internal nitrogen quota, a self-inhibition exponent, biomass-dependent maintenance and ammonium in the charge balance.

## Repository layout

| Folder | Content |
|---|---|
| [`Experiments_and_package/`](Experiments_and_package/) | Current version: the analysis package, both models, notebooks, data and campaign results. **Start here.** |
| [`Experiments_2025_and_previous/`](Experiments_2025_and_previous/) | Earlier code, notebooks and runs (including those behind the 2025 paper), kept for reference. |
| `IEEE ICA-ACCA 2025 article/` | 2025 paper and supplementary material (PDF). |
| `IEEE ICA-ACCA 2026 article/` | 2026 paper and supplementary material (PDF). |

## Quick start

```bash
cd Experiments_and_package
python -m venv .venv && source .venv/bin/activate     # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

Run notebooks and scripts from `Experiments_and_package` (the data file is read from the current directory). Full documentation of the toolkit, the models and the workflow is in [`Experiments_and_package/README_package.md`](Experiments_and_package/README_package.md).

## Papers

- Díaz-Figueroa et al. (2025). *Sensitivity Driven Optimization of a Batch Cultivation DAE Model of Mycobacterium smegmatis.* IEEE ICA-ACCA 2025.
- Díaz-Figueroa et al. (2026). *A Droop Quota DAE Model for Mycobacterium smegmatis Batch Cultivation Under Nutrient-Replete and Nitrogen-Limited Conditions.* IEEE ICA-ACCA 2026.

Experimental data: Apiyo et al. (2022), in `Experiments_and_package/Experimental_data.xlsx`.

## AI usage

Parts of this repository were developed with assistance from Claude (Anthropic), including code refactoring, documentation (READMEs) and repository organization. The modeling, experimental data, analysis design and scientific conclusions are the authors' own. All AI-assisted code and text were reviewed by the authors.
