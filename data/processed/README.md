# `pairs.json` — what this is and isn't

`pairs.json` in this folder is a **small illustrative fixture** (12 real
`(customer_message, brand_reply)` pairs, pulled from an actual AppleSupport
subsample during development) — it lets `--eval-only` run without a Kaggle
download so you can sanity-check the pipeline's plumbing quickly.

**It is NOT the ~2,000-3,000 pair subsample the report's headline numbers are
based on.** With only 12 pairs, the reply-drafting RAG step (TF-IDF retrieval
over historical pairs) will retrieve thin, often-irrelevant matches, and reply
quality will look noticeably worse than `REPORT.md`'s numbers as a result.

`run_pipeline.py` detects this automatically and logs a `[DEMO DATA]` warning
whenever `--eval-only` is run against fewer than 200 cached pairs.

## To regenerate a real cache

```bash
# Needs a Kaggle account + API token at ~/.kaggle/kaggle.json
# (see https://www.kaggle.com/docs/api for how to create one)
python run_pipeline.py --max-conversations 5000
```

This runs the full pipeline once (data_loader → thread_builder), which
downloads the dataset via `kagglehub`, filters to `AppleSupport`, reconstructs
threads, and overwrites `data/processed/pairs.json` with up to 5,000 real
pairs (see `run_pipeline.py::step_2_build_threads`, which caps the saved file
at 5,000 pairs for size). After that, `--eval-only` runs will use the real
cache instead of this fixture.
