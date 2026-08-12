# PNAS experiment submission package

This package contains the code, datasets, source data, and archived experiment outputs used for the submitted study. All paths in the maintained entry points are resolved relative to this directory; no workstation-specific path is required.

## Directory layout

- `code/eoh/`: evolutionary heuristic implementation and LLM interface.
- `code/analysis/`: baseline evaluation, transfer evaluation, scale analysis, and robustness scripts.
- `dataset/`: real and synthetic graph inputs.
- `evolve_experiment/evolution/`: archived experiment outputs and population records.
- `evolve_experiment/figures/`: figure-generation scripts and source data.
- `evolve_experiment/processed/`: processed analysis tables.
- `evolve_experiment/transfer/`: transferred or selected heuristic records used by downstream analyses.

## Environment

Python 3.10 or newer is recommended.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `.venv/bin/activate` or invoke its Python executable directly.

## LLM configuration

No API credential or private endpoint is included. Set both variables before starting an LLM-backed evolution run:

```text
EOH_API_ENDPOINT=<provider-endpoint>
EOH_API_KEY=<provider-key>
```

The endpoint and key can also be supplied with `--api_endpoint` and `--api_key`. Do not commit credentials or private infrastructure addresses.

## Main entry points

Run commands from this submission directory.

```text
python code/eoh/runEoH.py --help
python code/analysis/run_cn_baselines.py --help
python code/analysis/evaluate_synthetic_scales.py --help
python code/analysis/robustness/make_enron_robustness_figure.py --help
python code/analysis/robustness/make_sbm_robustness_figure.py --help
```

Figure scripts are stored beside their source-data CSV files under `evolve_experiment/figures/`. Generated figures are written back to the corresponding figure directory.

The archived `generate_metric_tables.py` aggregation workflow requires the detailed baseline-result directories used during analysis. The semantic-basin reanalysis script likewise references two adaptive-run directories that are represented in this package by derived source-data CSV files rather than complete intermediate run folders. The reviewer package includes the processed tables and figure source data needed to inspect the reported outputs.

## Reproducibility and provenance

- Random seeds used by the maintained workflows are defined in the relevant scripts.
- The bundled EoH-derived code is covered by `code/eoh/LICENSE`.
- Provenance and redistribution terms for third-party real-network datasets could not be established from files in the supplied workspace. Verify the original source, citation, version, and redistribution permission for every file in `dataset/real/` before public release. See `DATASET_SOURCES.txt`.

