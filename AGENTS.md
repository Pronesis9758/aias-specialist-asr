# AI Specialist ASR project instructions

## Mission

Build a reproducible manufacturing-domain Korean ASR pipeline. Every material result must be
traceable to an immutable run directory and the experiment registry.

## Required workflow

1. Use Python 3.12 and `uv`.
2. Run `uv sync --extra dev` after dependency changes.
3. Run `uv run pytest` and `uv run ruff check .` before committing.
4. Use `uv run aias run --config <path>` for pipeline validation.
5. Pin Hugging Face models to a resolved commit in `models/model-lock.yaml` before reporting
   non-fixture results.

## Data and security

- Never commit real employee/customer voice, confidential factory logs, secrets, tokens, model
  weights, or generated checkpoints.
- Public/fixture data may live under `data/sample/`; private inputs belong under `data/private/`.
- Colab may only receive public, synthetic, or approved de-identified data.
- Record data provenance and consent/approval status in the input manifest.

## Artifact contract

Each run must create `artifacts/runs/<run_id>/` containing at least:

- `config.snapshot.yaml`
- `environment.json`
- `prepared_manifest.csv`
- `predictions_baseline.csv`
- `predictions_corrected.csv`
- `metrics.json`
- `run_summary.md`
- `reports/evaluation_report.docx`

The same run must be registered in `backdata/experiments.sqlite3`.

## Human review gates

Do not claim production readiness without human verification of transcript labels, domain terms,
privacy approval, model-selection trade-offs, and final report conclusions.
