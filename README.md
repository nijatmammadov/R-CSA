# R-CSA

Code and data for my master's thesis *Reasoning Level Hallucination Detection in Language Models via
Counterfactual Sensitivity Analysis* (ELTE).

R-CSA (Reasoning-level Counterfactual Sensitivity Analysis) looks for the first wrong step in a
model's chain of thought, using only the model that wrote the chain. It extends the counterfactual
probing of Feng (2025, [arXiv:2508.01862](https://arxiv.org/abs/2508.01862)) from single statements
to the steps of a reasoning chain. For each step the model judges its own step against counterfactual
probes of it, and against versions of the question with one number changed. The first step whose
score falls below a threshold is reported as the error.

## Datasets

| Benchmark | Dev set (settings chosen here) | Test set (run once) | Models |
|---|---|---|---|
| GSM8K | 200 questions from the GSM8K train split | 400 questions from GSM8K-Platinum | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 |
| MATH Level 5 | 200 Level 5 questions from the MATH train split | 400 Level 5 questions from the MATH test split, excluding MATH-500 | Qwen2.5-7B-Instruct |

Early method choices (step shortening, probe types) were made on two code-generated sets of 100
word problems each, built from templates with known answers and reference steps
(`src/make_synthetic.py`, `data/synthetic`). They were used only for development, not for testing.

Each model solves the questions itself and judges its own steps. First-error labels were proposed
by a script and an LLM assistant, then confirmed by hand. The question files are in `data/dev` and
`data/test`, the solved and labelled chains in `data/dev` and `results/test`.

## Results

All settings were fixed on the dev sets before the test run. F1 is detection of wrong chains;
PB F1 is the ProcessBench metric for locating the first error.

| Method | Llama F1 | Llama PB F1 | Mistral F1 | Mistral PB F1 | Qwen F1 | Qwen PB F1 |
|---|---|---|---|---|---|---|
| R-CSA (ours) | 0.46 | 0.40 | 0.67 | 0.39 | 0.73 | 0.30 |
| Feng (2025) | 0.29 | - | 0.60 | - | 0.68 | - |
| Verbal confidence | 0.29 | - | 0.62 | - | 0.65 | - |
| SelfCheckGPT | 0.62 | - | 0.80 | - | 0.84 | - |
| ProcessBench critic | 0.29 | 0.16 | 0.59 | 0.26 | 0.63 | 0.24 |

R-CSA detects wrong chains better than Feng's method and verbal confidence on all three models, and
locates the first error better than the same model used as a ProcessBench critic (significant on
GSM8K, not on MATH Level 5). SelfCheckGPT detects better but cannot locate the error and needs about
twice as many calls per chain.

## Usage

```bash
pip install -r requirements.txt
cp env.example .env   # add your OpenRouter key
python -m pytest -q tests
```

Recompute the Llama dev table from the saved runs (no API calls):

```bash
python src/css_eval.py --labels data/dev/dev_llama31-8b_solved.json \
    --css results/dev/dev_llama31-8b_css2.json --feng results/dev/dev_llama31-8b_feng.json \
    --critic results/dev/dev_llama31-8b_critic.json --ptrue-full results/dev/dev_llama31-8b_ptrue_full.json \
    --cfq results/dev/dev_llama31-8b_cfq.json --ablations --frozen results/dev/css_frozen_llama31-8b.json
```

Pipeline: `pb_solve.py` (solve) → `css.py`, `css_cfq.py`, `pb_ptrue_full.py` (R-CSA stages) →
`css_eval.py` (score). Baselines: `pb_feng.py`, `pb_critic.py`, `pb_selfcheck.py`.
`notebooks/kaggle_vllm_run.ipynb` runs everything with an open model on Kaggle.
