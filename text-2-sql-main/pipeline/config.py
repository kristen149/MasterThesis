"""Run configuration — typed container for all CLI arguments.

Every module in the pipeline receives this object instead of reaching into
``argparse.Namespace`` directly. Keeps the runner stupid and the contract
between CLI and pipeline explicit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RunConfig:
    """All knobs that control a single benchmarking run."""

    # ---- what to execute --------------------------------------------------
    execution: str = "result_analysis"   # generate | evaluate | result_analysis (= both)
    language: str = "english"            # english | portuguese
    perspective: str = "sql"             # sql | process_mining | nlp (stored for metadata)
    metric: str = "EX"                   # EM | EX | both
    limit: Optional[int] = None          # None = full dataset

    # ---- prompt strategy --------------------------------------------------
    prompt: str = "zeroshot"             # zeroshot | lexicon | lf | lf-lite | lexicon-lf
    n_shots: int = 1                     # used by 1-shot / few-shot variants
    rag_top_k: int = 3
    use_prebuilt_prompt: bool = False    # if True and prompt=0-shot, use questions.json prompts verbatim

    # ---- backend selection ------------------------------------------------
    source: str = "local"                # local | hf
    model: str = "qwopus_glm_18b_healed_q4_k_m"

    # local API specifics
    api_url: str = "http://10.204.18.32:8080/v1/chat/completions"
    username: Optional[str] = None
    password: Optional[str] = None

    # HF specifics
    hf_token: Optional[str] = None
    hf_mode: str = "api"                 # api | local

    # ---- generation params ------------------------------------------------
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout: int = 300            # seconds
    request_retries: int = 3

    # ---- robustness -------------------------------------------------------
    on_error: str = "skip"               # skip | abort
    sql_exec_timeout: int = 5            # seconds per SQL query
    resume: bool = False

    # ---- output -----------------------------------------------------------
    results_root: str = "results"
    run_tag: Optional[str] = None        # optional manual disambiguator

    # ---- data paths (resolved from language) ------------------------------
    data_root: str = "data"

    def __post_init__(self) -> None:
        # Resolve env-var fallbacks for secrets so users don't put them on the CLI.
        if self.username is None:
            self.username = os.environ.get("T2S_API_USERNAME")
        if self.password is None:
            self.password = os.environ.get("T2S_API_PASSWORD")
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # ---- derived paths ----------------------------------------------------
    @property
    def lang_data_dir(self) -> str:
        return os.path.join(self.data_root, "dataset", self.language)

    @property
    def prompt_data_dir(self) -> str:
        return os.path.join(self.data_root, "prompt", self.language)

    @property
    def questions_path(self) -> str:
        return os.path.join(self.prompt_data_dir, "questions.json")

    @property
    def dev_path(self) -> str:
        return os.path.join(self.lang_data_dir, "dev.json")

    @property
    def gold_path(self) -> str:
        return os.path.join(self.lang_data_dir, "gold.txt")

    @property
    def db_path(self) -> str:
        return os.path.join(self.lang_data_dir, "event_log.sqlite")

    @property
    def exemplars_path(self) -> str:
        return os.path.join(self.prompt_data_dir, "exemplars.json")

    @property
    def rag_corpus_path(self) -> str:
        return os.path.join(self.prompt_data_dir, "rag_corpus.json")

    @property
    def hardness_tsv_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "text2sql4pm.tsv")

    def to_dict(self) -> dict:
        """Serializable view (secrets redacted)."""
        d = asdict(self)
        if d.get("password"):
            d["password"] = "***REDACTED***"
        if d.get("hf_token"):
            d["hf_token"] = "***REDACTED***"
        return d
