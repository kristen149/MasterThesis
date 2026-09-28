# Text-to-SQL Benchmarking & Analysis Pipeline

A modular, reproducible Text-to-SQL evaluation engine and statistical analysis toolkit built for evaluating Large Language Models (LLMs) on process mining Text-to-SQL benchmarks (such as the `Text2SQL4PM` dataset).

---

## 1. Project Directory & File Structure

```
text-2-sql-main/
├── PIPELINE.md                     # Comprehensive pipeline & reproduction documentation (this file)
├── README.md                       # High-level project summary
├── main.py                         # Primary entry point for generation, evaluation, and end-to-end benchmarking
├── corpus_analysis.py              # Reproducible corpus analysis (TTR, MTLD, SQL-grounded lexicalization)
├── requirements.txt                # Project dependencies
│
├── pipeline/                       # Modular Text-to-SQL Benchmarking Engine
│   ├── __init__.py
│   ├── cli.py                      # CLI argument parsing & typed RunConfig builder
│   ├── config.py                   # Dataclass configuration definitions
│   ├── data_loader.py              # Data loader for dev.json, gold.txt, tsv, & SQLite database
│   ├── executor.py                 # Safe SQLite query execution engine with timeouts
│   ├── naming.py                   # Standardized experiment naming & directory paths manager
│   ├── postprocess.py              # SQL extraction, cleaning, and normalization
│   ├── runner.py                   # Pipeline workflow orchestrator (full, generate, evaluate)
│   ├── writer.py                   # Streaming output writer (predictions.jsonl, run_meta.json, TSVs, txt)
│   ├── adapters/                   # Pluggable LLM Inference Adapters
│   │   ├── base.py                 # Abstract base class ModelAdapter
│   │   ├── local_api.py            # Local OpenAI-compatible API adapter (vLLM, Ollama, llama.cpp)
│   │   └── hf.py                   # HuggingFace Inference API & Transformers local adapter
│   ├── eval/                       # Pluggable Evaluation Metrics
│   │   ├── base.py                 # Abstract base class Evaluator
│   │   ├── exact_match.py          # Exact Match (EM) metric calculator
│   │   └── execution.py            # Execution Accuracy (EX) metric calculator
│   └── prompts/                    # Modular Prompt Engineering & Guidance Strategies
│       ├── base.py                 # Abstract base class PromptStrategy
│       ├── zero_shot.py            # Classic 0-shot prompt builder
│       ├── one_shot.py             # Few-shot prompt builder with exemplars
│       ├── cot.py                  # Chain-of-Thought (CoT) prompt builder
│       └── experiment.py           # Factorial ablation prompts: zeroshot, lexicon, lf, lexicon-lf
│
├── analysis/                       # Statistical Significance & Factorial Analysis
│   ├── statistical_analysis.py     # McNemar's test, 10k paired bootstrap CIs, DiD, GLMM engine
│   └── results/                    # Generated statistical JSON outputs (e.g., statistical_results_EX.json)
│
├── corpus_analysis_results/        # Output directory for corpus statistics (TTR, MTLD, CSV/JSON summaries)
│
├── scripts/                        # Robustness Evaluation & Visualization Utilities
│   ├── calculate_paraphrase_robustness.py # Paraphrase group consistency and robustness metrics
│   ├── dataset_analysis.py         # Secondary dataset exploration script
│   ├── graphs.py                   # Graph generation for accuracy breakdown by difficulty/category
│   ├── loader_results.py           # Loader interface for legacy result structures
│   ├── qualifiers.py               # SQL complexity and domain qualifier tagging logic
│   ├── results_analysis.py         # Qualifier-based performance summary breakdown
│   └── results_grouped_by_utterance.py # Per-utterance group aggregation
│
├── data/                           # Benchmark Dataset & Prompt Exemplars
│   ├── dataset/
│   │   ├── text2sql4pm.tsv         # Master Text2SQL4PM dataset (1655 English utterances)
│   │   ├── tables.json             # Database schema metadata
│   │   └── english/
│   │       ├── dev.json            # Evaluation splits
│   │       ├── gold.txt            # Ground-truth SQL references
│   │       └── event_log.sqlite    # SQLite event log database for EX evaluation
│   └── prompt/
│       └── english/
│           └── exemplars.json      # In-context exemplars for few-shot prompting
│
└── results/                        # Generated Benchmarking Outputs
    └── english/
        └── oss_<model_name>_<prompt_strategy>/
            ├── run_meta.json       # Execution metadata (git SHA, parameters, timestamp)
            ├── predictions.jsonl   # Per-sample predictions log
            ├── RESULTS_MODEL-*.txt # Gold-aligned prediction output
            └── evaluations/        # Tab-separated evaluation score files (EM / EX)
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.9+ (Python 3.10+ recommended)
- SQLite3

### Installation

```bash
# Clone repository and enter directory
cd text-2-sql-main

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Set authentication credentials depending on your chosen backend:

**For Local API Endpoints (e.g. OpenAI-compatible vLLM / server):**
```bash
# Linux / macOS
export T2S_API_USERNAME="YourUsername"
export T2S_API_PASSWORD="YourPassword"

# Windows PowerShell
$env:T2S_API_USERNAME="YourUsername"
$env:T2S_API_PASSWORD="YourPassword"
```

**For HuggingFace API:**
```bash
# Linux / macOS
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 3. Running the Text-to-SQL Pipeline (`main.py`)

`main.py` is the central CLI for executing inference, evaluation, and end-to-end benchmarking runs.

### CLI Parameters Quick Reference

| Argument | Choices / Default | Description |
| :--- | :--- | :--- |
| `--execution` | `result_analysis` (default), `generate`, `evaluate` | `generate` = model predictions only;<br>`evaluate` = score existing predictions;<br>`result_analysis` = generate + score end-to-end |
| `--source` | `local`, `hf` (**required**) | Backend type |
| `--model` | string (**required**) | Model identifier (HF repo ID or display name for local API) |
| `--api_url` | URL string | API endpoint URL (default: `http://10.204.18.32:8080/v1/chat/completions`) |
| `--prompt` | `zeroshot` (default), `lexicon`, `lf`, `lf-lite`, `lexicon-lf` (or `0-shot` alias) | Prompt strategy guidance strategy |
| `--metric` | `EX`, `EM`, `both` (default: `EX`) | Evaluation metric(s) to compute |
| `--language` | `english`, `portuguese` (default: `english`) | Dataset language split |
| `--limit` | integer | Cap maximum number of samples evaluated |
| `--temperature` | float (default: `0.0`) | Generation sampling temperature |
| `--max_tokens` | integer (default: `2000`) | Max generation tokens |
| `--sql_exec_timeout` | integer (default: `5`) | Timeout per SQL query execution in seconds |
| `--resume` | flag | Skip samples already written to `predictions.jsonl` |
| `--run_tag` | string | Optional tag appended to result folder name |

---

### Execution Examples

#### 1. End-to-End Run via Local API (OpenAI-compatible)
Runs generation + evaluation on 100 samples using execution accuracy (`EX`):
```bash
python main.py \
  --source local \
  --api_url http://10.204.18.32:8080/v1/chat/completions \
  --model qwopus_glm_18b_healed_q4_k_m \
  --execution result_analysis \
  --prompt zeroshot \
  --metric EX \
  --limit 100
```

#### 2. Factorial Experiment Ablations
Execute specific prompt guidance strategies (`zeroshot`, `lexicon`, `lf`, `lexicon-lf`):
```bash
# Lexicon-Guided Strategy
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --prompt lexicon \
  --metric EX

# Combined (Lexicon + Logical-Form) Strategy
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --prompt lexicon-lf \
  --metric EX
```

#### 3. HuggingFace Inference API Run
```bash
python main.py \
  --source hf \
  --model meta-llama/Llama-3-8B-Instruct \
  --execution result_analysis \
  --prompt cot \
  --metric both \
  --limit 50
```

#### 4. Resume an Interrupted / Crashed Run
Use `--resume` to skip samples that were already saved in `predictions.jsonl`:
```bash
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --prompt 0-shot \
  --metric EX \
  --resume
```

#### 5. Re-Score Existing Predictions (No Model Calls)
Re-evaluate existing `predictions.jsonl` output files using both EM and EX:
```bash
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --execution evaluate \
  --prompt 0-shot \
  --metric both
```

---

## 4. Running Corpus Analysis (`corpus_analysis.py`)

`corpus_analysis.py` performs descriptive, reproducible analysis of the natural language queries and SQL targets in the `Text2SQL4PM` dataset.

### Scope & Metrics
- **Dataset Validation**: Verifies utterance count (1,655) and paraphrase group count (205).
- **Lexical Diversity**: Computes Type-Token Ratio (TTR) and Measure of Textual Lexical Diversity (MTLD).
- **Paraphrase Similarity**: Calculates surface paraphrase similarity (Jaccard distance anchored to reference `_0`).
- **SQL-Grounded Categorization**: Evaluates column, value, and operation lexicalization in NL queries.

### How to Run
```bash
python corpus_analysis.py
```
Outputs are saved directly into the `corpus_analysis_results/` directory as structured JSON and CSV files.

---

## 5. Running Statistical Analysis (`analysis/statistical_analysis.py`)

`analysis/statistical_analysis.py` performs hypothesis testing for 2x2 factorial prompt-guidance experiments.

### Research Questions Addressed
- **RQ1**: Zero-shot vs. Lexical Guidance (McNemar's test + Paired Bootstrap 95% CI)
- **RQ2**: Zero-shot vs. Logical-Form Guidance (McNemar's test + Paired Bootstrap 95% CI)
- **RQ3a**: Lexicon / LF vs. Combined Guidance (McNemar's test + Paired Bootstrap 95% CI)
- **RQ3b**: Lexical $\times$ Logical-Form Interaction (Bootstrap Difference-in-Differences & GLMM)

### How to Run
```bash
python analysis/statistical_analysis.py
```
Statistical results are exported to `analysis/results/statistical_results_EX.json` (or `EM.json`).

---

## 6. Running Auxiliary & Visualization Scripts (`scripts/`)

### Paraphrase Robustness
Calculate performance consistency across paraphrases within each group:
```bash
python scripts/calculate_paraphrase_robustness.py
```

### Results Analysis & Tagging
Run qualifier breakdown analysis across SQL complexity tiers:
```bash
python scripts/results_analysis.py
```

### Graph Generation
Generate visualization plots:
```bash
python scripts/graphs.py
```

---

## 7. Results Output Structure

All pipeline outputs are automatically saved under `results/<language>/<experiment_folder>/`:

```
results/english/oss_qwopus_glm_18b_healed_q4_k_m_0-shot/
├── run_meta.json                                # Config details, git SHA, dataset hash, timestamp
├── predictions.jsonl                                # Detailed JSONL log (prompt, raw response, extracted SQL, exec status)
├── RESULTS_MODEL-qwopus_glm_18b_healed_q4_k_m.txt   # Gold-aligned prediction SQL file
└── evaluations/
    ├── scores_opr_qwopus_glm_18b_healed_q4_k_m_EM.tsv # Per-item EM score TSV
    └── scores_opr_qwopus_glm_18b_healed_q4_k_m_EX.tsv # Per-item EX score TSV
```

---

## 8. Extending the Pipeline

The architecture is built on abstract base classes, making extension seamless:

1. **Add a New Prompt Strategy**:
   Create a new file in `pipeline/prompts/my_strategy.py` subclassing `PromptStrategy`. Register it in `pipeline/prompts/__init__.py` and add the option to `pipeline/cli.py`.

2. **Add a New Backend Adapter**:
   Create `pipeline/adapters/my_adapter.py` subclassing `ModelAdapter`. Register it in `pipeline/adapters/__init__.py`.

3. **Add a New Metric**:
   Create `pipeline/eval/my_metric.py` subclassing `Evaluator`. Register it in `pipeline/eval/__init__.py`.
