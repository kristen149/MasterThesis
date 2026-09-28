# Master Thesis: Open-Source Large Language Models for Natural Language Queries in Process Mining
**Capabilities, Evaluation & Perspectives**

A modular, reproducible Text-to-SQL evaluation engine, execution environment, and statistical analysis toolkit built for evaluating Large Language Models (LLMs) on process mining domains.

---

## Paper & Repository References

* Research Paper: [Text-to-SQL Oriented to the Process Mining Domain: A PT-EN Dataset for Query Translation](https://arxiv.org/pdf/2509.09684)
* Original Repository: [https://github.com/pm-usp/text-2-sql](https://github.com/pm-usp/text-2-sql)

---

## Project Overview

This repository hosts an automated Text-to-SQL benchmarking and execution framework. Rather than performing static script runs, the framework provides an end-to-end modular pipeline that handles LLM inference, schema/domain prompt engineering, sandboxed SQL query execution against process mining event log databases, dual-metric evaluation, corpus linguistic diversity profiling, and statistical hypothesis testing.

---

## What the Pipeline Executes

The execution framework is composed of five core modules:

### 1. LLM Inference & Prompt Strategy Engine (`main.py` & `pipeline/`)
* **Pluggable Backend Adapters (`pipeline/adapters/`)**:
  * `local`: OpenAI-compatible local API server (e.g. vLLM, Ollama, llama.cpp with authentication).
  * `hf`: HuggingFace Inference API & local `transformers` model runner.
* **Factorial Prompt Guidance Strategies (`pipeline/prompts/`)**:
  * `zeroshot`: Baseline zero-shot schema prompt.
  * `one_shot` / `cot`: Exemplar-based few-shot and Chain-of-Thought reasoning.
  * `lexicon`: Lexicon-guided entity representation.
  * `lf` / `lf-lite`: Logical-Form schema representation.
  * `lexicon-lf`: Combined 2x2 factorial ablation strategy (Lexicon + Logical-Form).
* **Execution Modes (`--execution`)**:
  * `result_analysis`: End-to-end mode running model generation followed immediately by evaluation.
  * `generate`: Model inference mode writing predictions to stream logs.
  * `evaluate`: Re-scores pre-existing prediction files without invoking model APIs.
* **Resiliency & Streaming**:
  * Real-time streaming to `predictions.jsonl` and metadata logging in `run_meta.json`.
  * Support for `--resume` flag to recover interrupted runs without re-evaluating completed samples.

### 2. Safe SQL Execution & Evaluation Engine (`pipeline/executor.py` & `pipeline/eval/`)
* **SQL Normalization (`pipeline/postprocess.py`)**: Automatically strips Markdown code blocks, cleans syntax errors, and normalizes output queries.
* **Sandboxed Query Execution (`pipeline/executor.py`)**: Executes generated SQL queries against target SQLite process mining databases (`event_log.sqlite`) with per-query timeout guards (`--sql_exec_timeout`).
* **Evaluation Metrics (`pipeline/eval/`)**:
  * **Exact Match (EM)**: Structural AST / string exact matching against ground-truth queries.
  * **Execution Accuracy (EX)**: Comparing query execution result sets against gold reference execution outputs.

### 3. Statistical Analysis & Hypothesis Testing (`analysis/statistical_analysis.py`)
* **Hypothesis Evaluation**: Evaluates 2x2 factorial prompt-guidance interaction effects across research questions:
  * **RQ1**: Zero-shot vs. Lexical Guidance
  * **RQ2**: Zero-shot vs. Logical-Form Guidance
  * **RQ3a**: Single vs. Combined Guidance
  * **RQ3b**: Lexical $\times$ Logical-Form Interaction
* **Statistical Methods**:
  * **McNemar's Test**: Non-parametric test for paired binary outcomes.
  * **Paired Bootstrap Confidence Intervals**: 10,000-iteration paired bootstrapping for robust 95% CIs.
  * **Interaction Models**: Difference-in-Differences (DiD) and Generalized Linear Mixed Models (GLMM).

### 4. Corpus & Linguistic Diversity Profiling (`corpus_analysis.py`)
* **Lexical Diversity**: Computes Type-Token Ratio (TTR) and Measure of Textual Lexical Diversity (MTLD) across English and Portuguese queries.
* **Paraphrase Similarity**: Surface Jaccard distance calculation across reference paraphrase groups.
* **SQL-Grounded Profiling**: Analyzes column name, value, and SQL operation lexicalization within natural language utterances.

### 5. Paraphrase Robustness & Analytics (`scripts/`)
* **Group Consistency (`calculate_paraphrase_robustness.py`)**: Computes accuracy consistency across paraphrase groups.
* **Qualifier & Complexity Breakdown (`results_analysis.py`)**: Aggregates model performance across SQL complexity tiers (Easy, Medium, Hard, Extra) and domain qualifiers.
* **Graph Generation (`graphs.py`)**: Automatically produces comparative charts and performance figures.

---

## Directory Structure

```
.
├── main.py                             # Primary CLI entry point for generation & evaluation
├── corpus_analysis.py                  # Corpus linguistic diversity & lexicalization analyzer
├── README.md                           # Project documentation (this file)
├── PIPELINE.md                         # Detailed pipeline technical specifications
├── requirements.txt                    # Python dependencies
│
├── pipeline/                           # Modular Text-to-SQL Benchmarking Engine
│   ├── cli.py                          # Command-line argument parser & configuration builder
│   ├── config.py                       # Typed dataclass definitions
│   ├── data_loader.py                  # Dev sets, gold SQL, TSV, and SQLite data loaders
│   ├── executor.py                     # Sandboxed SQLite execution engine with timeouts
│   ├── naming.py                       # Standardized directory naming conventions
│   ├── postprocess.py                  # SQL cleaning and extraction utilities
│   ├── runner.py                       # Pipeline workflow orchestrator
│   ├── writer.py                       # Output streamer (predictions.jsonl, run_meta.json, TSVs)
│   ├── adapters/                       # Model inference adapters (Local API, HuggingFace)
│   ├── eval/                           # Metric calculators (EM, EX)
│   └── prompts/                        # Factorial prompt strategies (zeroshot, lexicon, lf, etc.)
│
├── analysis/                           # Statistical Significance Engine
│   ├── statistical_analysis.py         # McNemar, Bootstrap CIs, DiD, and GLMM testing engine
│   └── results/                        # Exported statistical JSON outputs
│
├── corpus_analysis_results/            # Output folder for TTR, MTLD, and corpus metrics
│
├── scripts/                            # Analysis & Visualization Utilities
│   ├── calculate_paraphrase_robustness.py
│   ├── dataset_analysis.py
│   ├── graphs.py
│   ├── loader_results.py
│   ├── qualifiers.py
│   ├── results_analysis.py
│   └── results_grouped_by_utterance.py
│
├── data/                               # Datasets & Database Schemas
│   └── dataset/
│       ├── text2sql4pm.tsv
│       ├── tables.json
│       └── english/
│           ├── dev.json
│           ├── gold.txt
│           └── event_log.sqlite
│
└── results/                            # Output directory for benchmark run logs
```

---

## Environment Setup

### Prerequisites
* Python 3.9+ (Python 3.10+ recommended)
* SQLite3

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
Set credentials depending on your chosen inference backend:

**Local OpenAI-Compatible API (vLLM / server):**
```bash
# Windows PowerShell
$env:T2S_API_USERNAME="YourUsername"
$env:T2S_API_PASSWORD="YourPassword"

# Linux / macOS
export T2S_API_USERNAME="YourUsername"
export T2S_API_PASSWORD="YourPassword"
```

**HuggingFace API:**
```bash
# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Linux / macOS
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## How to Run

### 1. End-to-End Benchmarking (`main.py`)

Run generation and evaluation end-to-end on local API endpoint:
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

### 2. Factorial Prompt Strategy Experiments

Run specific prompt guidance strategies (`zeroshot`, `lexicon`, `lf`, `lexicon-lf`):
```bash
# Combined Lexicon + Logical-Form Guidance Strategy
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --prompt lexicon-lf \
  --metric EX
```

### 3. Resuming Interrupted Runs & Re-scoring

**Resume an interrupted run:**
```bash
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --prompt 0-shot \
  --metric EX \
  --resume
```

**Re-score existing predictions without model calls:**
```bash
python main.py \
  --source local \
  --model qwopus_glm_18b_healed_q4_k_m \
  --execution evaluate \
  --prompt 0-shot \
  --metric both
```

### 4. Corpus Linguistic Diversity Analysis
```bash
python corpus_analysis.py
```

### 5. Statistical Hypothesis Testing
```bash
python analysis/statistical_analysis.py
```

### 6. Paraphrase Robustness & Visualizations
```bash
# Calculate paraphrase robustness metrics
python scripts/calculate_paraphrase_robustness.py

# Analyze qualifier performance breakdowns
python scripts/results_analysis.py

# Generate comparison plots
python scripts/graphs.py
```
