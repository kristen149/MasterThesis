"""
statistical_analysis.py
========================
Statistical analysis for the Text-to-SQL prompt-guidance 2x2 factorial experiment.

Statistical unit: original matched TSV item IDs (no question-level collapse).
Each row in the TSV is treated as one paired observation, aligned on its full
run_id (e.g. "42_0", "42_1", ...). This matches the methodology of the old
significance.py and operates on all ~1655 items per condition.

Research Questions
------------------
RQ1  : Zero-shot vs. Lexical guidance          -> McNemar + paired-bootstrap CI
RQ2  : Zero-shot vs. Logical-form guidance     -> McNemar + paired-bootstrap CI
RQ3a : Lexicon vs. Combined                    -> McNemar + paired-bootstrap CI
       LF vs. Combined                         -> McNemar + paired-bootstrap CI
RQ3b : Lexical x Logical-form interaction      -> Bootstrap DiD + GLMM

Primary analysis uses item-level run_ids directly from the TSV.
Question-level (pass@k) statistics are computed for DESCRIPTIVE purposes only
and are clearly labelled as secondary. They do NOT feed into any hypothesis test.
"""

import os
import sys
import io
import math
import random
import warnings
import json
import csv
from collections import defaultdict

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import scipy.stats as stats

# Optional statsmodels for GLMM
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not found - GLMM (RQ3b) will be skipped.")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_EN  = os.path.join(BASE, "results", "english")
OUTPUT_DIR  = os.path.join(BASE, "analysis", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITION_DIRS = {
    "zeroshot" : "oss_qwopus_glm_18b_healed_q4_k_m_zeroshot",
    "lexicon"  : "oss_qwopus_glm_18b_healed_q4_k_m_lexicon",
    "lf"       : "oss_qwopus_glm_18b_healed_q4_k_m_lf",
    "combined" : "oss_qwopus_glm_18b_healed_q4_k_m_lexicon-lf",
}

SEED   = 0          # matches old significance.py
N_BOOT = 10_000
ALPHA  = 0.05
METRIC = "EX"       # "EX" or "EM"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def tsv_path(condition: str) -> str:
    folder = CONDITION_DIRS[condition]
    fname  = f"scores_opr_oss_qwopus_glm_18b_healed_q4_k_m_{METRIC}.tsv"
    return os.path.join(RESULTS_EN, folder, "evaluations", fname)


def load_scores(condition: str) -> pd.DataFrame:
    """
    Load the TSV for *condition* and return a DataFrame indexed by run_id.

    Columns: run_id (index), predicted, gold, difficulty, score, question_id
    The run_id (e.g. "42_3") is kept as-is and used as the primary item key
    for all hypothesis tests.  question_id is derived for descriptive use only.
    """
    path = tsv_path(condition)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["run_id", "predicted", "gold", "difficulty", "score"],
        dtype={"run_id": str},
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    df["score"]       = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["question_id"] = df["run_id"].str.rsplit("_", n=1).str[0]  # descriptive only
    df = df.set_index("run_id")
    df["condition"] = condition
    return df

# ---------------------------------------------------------------------------
# Primary item-level McNemar test
# (mirrors the old significance.py logic exactly)
# ---------------------------------------------------------------------------

def mcnemar_test(sA: pd.Series, sB: pd.Series) -> dict:
    """
    Item-level McNemar's test.

    sA, sB : binary score Series indexed by run_id (or any shared item ID).
    Computes the 2x2 contingency table on the intersection of IDs, then
    applies the exact binomial form when b+c < 25, otherwise the
    continuity-corrected chi-square approximation.
    """
    shared = sA.index.intersection(sB.index)
    yA = sA.loc[shared].astype(int)
    yB = sB.loc[shared].astype(int)
    n  = len(shared)

    a = int(((yA == 1) & (yB == 1)).sum())   # both correct
    b = int(((yA == 1) & (yB == 0)).sum())   # A correct, B wrong
    c = int(((yA == 0) & (yB == 1)).sum())   # A wrong,   B correct
    d = int(((yA == 0) & (yB == 0)).sum())   # both wrong

    nd = b + c   # discordant pairs
    if nd == 0:
        p, stat, method = 1.0, 0.0, "exact (no discordant pairs)"
    elif nd < 25:
        res             = stats.binomtest(min(b, c), nd, 0.5)
        p, stat, method = res.pvalue, float("nan"), "exact binomial McNemar"
    else:
        stat   = (abs(b - c) - 1) ** 2 / nd
        p      = 1 - stats.chi2.cdf(stat, df=1)
        method = "chi-squared with continuity correction"

    return {
        "n_items"      : n,
        "a"            : a, "b": b, "c": c, "d": d,
        "n_discordant" : nd,
        "statistic"    : stat,
        "p_value"      : p,
        "significant"  : bool(p < ALPHA),
        "method"       : method,
        "acc_A"        : float(yA.mean()),
        "acc_B"        : float(yB.mean()),
        "delta"        : float(yB.mean() - yA.mean()),
    }

# ---------------------------------------------------------------------------
# Primary item-level paired bootstrap CI
# (mirrors the old significance.py logic exactly)
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    sA: pd.Series,
    sB: pd.Series,
    n_boot: int = N_BOOT,
    alpha:  float = ALPHA,
    seed:   int   = SEED,
) -> dict:
    """
    Percentile bootstrap CI for delta = acc(B) - acc(A).
    Resamples the shared item IDs with replacement; the same draw is applied
    to both conditions to preserve pairing.
    """
    rng    = np.random.default_rng(seed)
    shared = sA.index.intersection(sB.index)
    vA     = sA.loc[shared].to_numpy(dtype=float)
    vB     = sB.loc[shared].to_numpy(dtype=float)
    n      = len(vA)

    obs_delta = vB.mean() - vA.mean()

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        deltas[i] = vB[idx].mean() - vA[idx].mean()

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Bootstrap p: shift to H0 (delta=0), two-sided
    shifted = deltas - obs_delta
    p_boot  = float(np.mean(np.abs(shifted) >= abs(obs_delta)))

    return {
        "obs_delta"   : float(obs_delta),
        "ci_lo"       : lo,
        "ci_hi"       : hi,
        "ci_level"    : f"{int((1-alpha)*100)}%",
        "p_boot"      : p_boot,
        "significant" : bool((lo > 0) or (hi < 0)),
    }

# ---------------------------------------------------------------------------
# Subgroup McNemar (per difficulty) — item-level
# ---------------------------------------------------------------------------

def subgroup_mcnemar(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
) -> dict:
    """
    McNemar + bootstrap per difficulty bucket, using item-level run_ids.
    dfA/dfB are DataFrames indexed by run_id with columns [score, difficulty].
    """
    results      = {}
    difficulties = sorted(dfA["difficulty"].unique())
    for diff in difficulties:
        sA = dfA.loc[dfA["difficulty"] == diff, "score"]
        sB = dfB.loc[dfB["difficulty"] == diff, "score"]
        shared = sA.index.intersection(sB.index)
        if len(shared) < 2:
            continue
        mc  = mcnemar_test(sA, sB)
        bci = paired_bootstrap_ci(sA, sB)
        results[diff] = {**mc, **{"boot_" + k: v for k, v in bci.items()}}
    return results

# ---------------------------------------------------------------------------
# RQ3b: item-level bootstrap Difference-in-Differences
# ---------------------------------------------------------------------------

def bootstrap_did(
    sZ: pd.Series,   # zeroshot
    sL: pd.Series,   # lexicon
    sF: pd.Series,   # lf
    sC: pd.Series,   # combined
    n_boot: int   = N_BOOT,
    alpha:  float = ALPHA,
    seed:   int   = SEED,
) -> dict:
    """
    Paired bootstrap DiD at item level.

    DiD = [acc(combined) - acc(lf)] - [acc(lexicon) - acc(zeroshot)]

    Intersects the full run_id set across all four conditions, then
    bootstraps those matched item IDs with replacement.
    """
    rng    = np.random.default_rng(seed)
    shared = (sZ.index
              .intersection(sL.index)
              .intersection(sF.index)
              .intersection(sC.index))

    vZ = sZ.loc[shared].to_numpy(dtype=float)
    vL = sL.loc[shared].to_numpy(dtype=float)
    vF = sF.loc[shared].to_numpy(dtype=float)
    vC = sC.loc[shared].to_numpy(dtype=float)
    n  = len(shared)

    def did(a0, al, af, ac):
        return (ac.mean() - af.mean()) - (al.mean() - a0.mean())

    obs_did   = did(vZ, vL, vF, vC)
    boot_dids = np.empty(n_boot)
    for i in range(n_boot):
        idx           = rng.integers(0, n, size=n)
        boot_dids[i]  = did(vZ[idx], vL[idx], vF[idx], vC[idx])

    lo = float(np.percentile(boot_dids, 100 * alpha / 2))
    hi = float(np.percentile(boot_dids, 100 * (1 - alpha / 2)))

    shifted = boot_dids - obs_did
    p_boot  = float(np.mean(np.abs(shifted) >= abs(obs_did)))

    return {
        "statistical_unit" : "original matched TSV item IDs (no question-level collapse)",
        "formula"          : "DiD = [acc(combined)-acc(lf)] - [acc(lexicon)-acc(zeroshot)]",
        "n_items"          : int(n),
        "obs_did"          : float(obs_did),
        "ci_lo"            : lo,
        "ci_hi"            : hi,
        "ci_level"         : f"{int((1-alpha)*100)}%",
        "p_boot"           : p_boot,
        "significant"      : bool((lo > 0) or (hi < 0)),
        "acc_zeroshot"     : float(vZ.mean()),
        "acc_lexicon"      : float(vL.mean()),
        "acc_lf"           : float(vF.mean()),
        "acc_combined"     : float(vC.mean()),
        "marginal_lex"     : float(vL.mean() - vZ.mean()),
        "marginal_lf"      : float(vF.mean() - vZ.mean()),
        "expected_add"     : float((vL.mean() - vZ.mean()) + (vF.mean() - vZ.mean())),
        "actual_combined_gain": float(vC.mean() - vZ.mean()),
    }

# ---------------------------------------------------------------------------
# GLMM (RQ3b secondary) — item level
# ---------------------------------------------------------------------------

def run_glmm(all_dfs: dict) -> dict:
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}

    cond_flags = {
        "zeroshot" : (0, 0),
        "lexicon"  : (1, 0),
        "lf"       : (0, 1),
        "combined" : (1, 1),
    }
    frames = []
    for cond, (lex_flag, lf_flag) in cond_flags.items():
        sub = all_dfs[cond].reset_index().copy()
        sub["lex"] = lex_flag
        sub["lf"]  = lf_flag
        frames.append(sub)

    df_model = pd.concat(frames, ignore_index=True)
    df_model["score"] = df_model["score"].astype(float)
    diff_map = {"easy": 0, "medium": 1, "hard": 2, "extra": 3, "no_hardness": 1}
    df_model["difficulty_num"] = df_model["difficulty"].map(diff_map).fillna(1)

    try:
        formula = "score ~ lex * lf + difficulty_num"
        md  = smf.mixedlm(formula, df_model, groups=df_model["question_id"])
        mdf = md.fit(method="lbfgs", reml=False)
        result = {
            "formula"     : formula,
            "n_obs"       : int(len(df_model)),
            "n_questions" : int(df_model["question_id"].nunique()),
            "converged"   : bool(mdf.converged),
            "params"      : {k: float(v) for k, v in mdf.params.items()},
            "pvalues"     : {k: float(v) for k, v in mdf.pvalues.items()},
            "conf_int"    : mdf.conf_int().to_dict(),
            "note"        : (
                "MixedLM (statsmodels) is a linear probability model approximation. "
                "For a true logistic GLMM use lme4/glmmTMB in R."
            ),
        }
    except Exception as exc:
        result = {"error": str(exc)}
    return result

# ---------------------------------------------------------------------------
# Descriptive question-level accuracy (secondary — does NOT feed tests)
# ---------------------------------------------------------------------------

def question_level_descriptive(dfs: dict) -> dict:
    """
    DESCRIPTIVE ONLY — not used in any hypothesis test.
    Collapses to unique question_ids using pass@k (max score across runs).
    """
    out = {}
    for cond, df in dfs.items():
        q = df.groupby("question_id")["score"].max()
        out[cond] = {
            "n_questions"  : int(len(q)),
            "pass_at_k_acc": float(q.mean()),
            "mean_run_acc" : float(df["score"].mean()),
        }
    return out

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    if p < 0.0001:
        return f"{p:.2e} ***"
    elif p < 0.001:
        return f"{p:.4f} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    elif p < 0.10:
        return f"{p:.4f} ."
    else:
        return f"{p:.4f}"


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def print_comparison(label: str, mc: dict, bci: dict) -> None:
    sig = "YES (p<0.05)" if mc["significant"] else "no"
    print(f"\n  Statistical unit  : original matched TSV item IDs (no question-level collapse)")
    print(f"  Comparison        : {label}")
    print(f"  N matched items   : {mc['n_items']}")
    print(f"  Accuracy A        : {mc['acc_A']:.4f}  ({mc['acc_A']*100:.2f}%)")
    print(f"  Accuracy B        : {mc['acc_B']:.4f}  ({mc['acc_B']*100:.2f}%)")
    print(f"  Delta (B - A)     : {mc['delta']:+.4f}  ({mc['delta']*100:+.2f} pp)")
    print(f"  Contingency       : a={mc['a']}  b={mc['b']}  c={mc['c']}  d={mc['d']}")
    print(f"  N discordant      : {mc['n_discordant']}")
    print(f"  McNemar method    : {mc['method']}")
    if not math.isnan(mc["statistic"]):
        print(f"  Chi2 statistic    : {mc['statistic']:.4f}")
    print(f"  p-value           : {fmt_p(mc['p_value'])}  [{sig}]")
    print(f"  {bci['ci_level']} bootstrap CI : [{bci['ci_lo']:+.4f}, {bci['ci_hi']:+.4f}]")
    print(f"  Bootstrap p       : {fmt_p(bci['p_boot'])}")


def print_subgroups(sg: dict) -> None:
    print(f"\n  {'Difficulty':<14} {'n items':>8} {'Acc A':>7} {'Acc B':>7} "
          f"{'Delta':>8}  {'p-value':>14}  {'95% CI':>22}  {'Sig?':>5}")
    print(f"  {'-'*88}")
    for diff, res in sg.items():
        sig    = "YES" if res["significant"] else "no"
        ci_str = f"[{res['boot_ci_lo']:+.4f}, {res['boot_ci_hi']:+.4f}]"
        print(f"  {diff:<14} {res['n_items']:>8} {res['acc_A']:>7.4f} {res['acc_B']:>7.4f} "
              f"{res['delta']:>+8.4f}  {fmt_p(res['p_value']):>14}  {ci_str:>22}  {sig:>5}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print(f"  Text-to-SQL Statistical Analysis  |  Metric: {METRIC}")
    print(f"  Statistical unit: original matched TSV item IDs (no question-level collapse)")
    print(f"  Bootstrap iterations: {N_BOOT:,}  |  Seed: {SEED}  |  Alpha: {ALPHA}")
    print("=" * 70)

    # ---- Load all conditions ------------------------------------------------
    print("\n[1/5] Loading data ...")
    dfs: dict[str, pd.DataFrame] = {}
    for cond in CONDITION_DIRS:
        path = tsv_path(cond)
        print(f"  {cond:<12}  ->  {os.path.relpath(path, BASE)}")
        dfs[cond] = load_scores(cond)
        n   = len(dfs[cond])
        acc = dfs[cond]["score"].mean()
        print(f"             N items={n}  item accuracy={acc:.4f}  ({acc*100:.2f}%)")

    # Primary score Series indexed by run_id (item ID)
    s: dict[str, pd.Series] = {cond: dfs[cond]["score"] for cond in dfs}

    # ---- Descriptive question-level stats (secondary, not used in tests) ----
    print("\n[2/5] Descriptive statistics (secondary — not used in hypothesis tests)")
    print("  [pass@k = max score across runs per question_id]")
    desc = question_level_descriptive(dfs)
    print(f"\n  {'Condition':<12}  {'N questions':>12}  {'Pass@k acc':>12}  {'Mean run acc':>14}")
    print(f"  {'-'*56}")
    for cond, d in desc.items():
        print(f"  {cond:<12}  {d['n_questions']:>12}  "
              f"{d['pass_at_k_acc']:>12.4f}  {d['mean_run_acc']:>14.4f}")

    all_results = {}

    # =========================================================================
    # RQ1: Zero-shot vs. Lexical guidance
    # =========================================================================
    section("RQ1: Zero-shot vs. Lexical guidance")
    print("\n[3a/5] Running item-level McNemar + bootstrap ...")
    mc_rq1  = mcnemar_test(s["zeroshot"], s["lexicon"])
    bci_rq1 = paired_bootstrap_ci(s["zeroshot"], s["lexicon"])
    print_comparison("Zero-shot (A) vs. Lexicon (B)", mc_rq1, bci_rq1)

    print("\n  --- Subgroup analysis by difficulty (item-level) ---")
    sg_rq1 = subgroup_mcnemar(dfs["zeroshot"], dfs["lexicon"])
    print_subgroups(sg_rq1)
    all_results["RQ1"] = {"mcnemar": mc_rq1, "bootstrap_ci": bci_rq1, "subgroups": sg_rq1}

    # =========================================================================
    # RQ2: Zero-shot vs. Logical-form guidance
    # =========================================================================
    section("RQ2: Zero-shot vs. Logical-form guidance")
    print("\n[3b/5] Running item-level McNemar + bootstrap ...")
    mc_rq2  = mcnemar_test(s["zeroshot"], s["lf"])
    bci_rq2 = paired_bootstrap_ci(s["zeroshot"], s["lf"])
    print_comparison("Zero-shot (A) vs. LF (B)", mc_rq2, bci_rq2)

    print("\n  --- Subgroup analysis by difficulty (item-level) ---")
    sg_rq2 = subgroup_mcnemar(dfs["zeroshot"], dfs["lf"])
    print_subgroups(sg_rq2)
    all_results["RQ2"] = {"mcnemar": mc_rq2, "bootstrap_ci": bci_rq2, "subgroups": sg_rq2}

    # =========================================================================
    # RQ3a: Combined vs. individual conditions
    # =========================================================================
    section("RQ3a: Lexicon vs. Combined")
    print("\n[3c/5] Running item-level McNemar + bootstrap ...")
    mc_3a_lex  = mcnemar_test(s["lexicon"], s["combined"])
    bci_3a_lex = paired_bootstrap_ci(s["lexicon"], s["combined"])
    print_comparison("Lexicon (A) vs. Combined (B)", mc_3a_lex, bci_3a_lex)
    sg_3a_lex = subgroup_mcnemar(dfs["lexicon"], dfs["combined"])
    print("\n  --- Subgroup analysis by difficulty (item-level) ---")
    print_subgroups(sg_3a_lex)

    section("RQ3a: Logical-form vs. Combined")
    mc_3a_lf  = mcnemar_test(s["lf"], s["combined"])
    bci_3a_lf = paired_bootstrap_ci(s["lf"], s["combined"])
    print_comparison("LF (A) vs. Combined (B)", mc_3a_lf, bci_3a_lf)
    sg_3a_lf = subgroup_mcnemar(dfs["lf"], dfs["combined"])
    print("\n  --- Subgroup analysis by difficulty (item-level) ---")
    print_subgroups(sg_3a_lf)

    all_results["RQ3a"] = {
        "lexicon_vs_combined": {
            "mcnemar": mc_3a_lex, "bootstrap_ci": bci_3a_lex, "subgroups": sg_3a_lex,
        },
        "lf_vs_combined": {
            "mcnemar": mc_3a_lf,  "bootstrap_ci": bci_3a_lf,  "subgroups": sg_3a_lf,
        },
    }

    # =========================================================================
    # RQ3b: Lexical x Logical-form interaction (item-level DiD + GLMM)
    # =========================================================================
    section("RQ3b: Lexical x LF Interaction (item-level DiD + GLMM)")
    print("\n[4/5] Running item-level DiD ...")

    did = bootstrap_did(s["zeroshot"], s["lexicon"], s["lf"], s["combined"])

    print(f"\n  Statistical unit  : {did['statistical_unit']}")
    print(f"  DiD formula       : {did['formula']}")
    print(f"  N matched items   : {did['n_items']}")
    print(f"\n  Item-level accuracies:")
    print(f"    Zero-shot  : {did['acc_zeroshot']:.4f}  ({did['acc_zeroshot']*100:.2f}%)")
    print(f"    Lexicon    : {did['acc_lexicon']:.4f}  ({did['acc_lexicon']*100:.2f}%)")
    print(f"    LF         : {did['acc_lf']:.4f}  ({did['acc_lf']*100:.2f}%)")
    print(f"    Combined   : {did['acc_combined']:.4f}  ({did['acc_combined']*100:.2f}%)")
    print(f"\n  Marginal effects (over zero-shot):")
    print(f"    Delta Lexicon          : {did['marginal_lex']:+.4f}  ({did['marginal_lex']*100:+.2f} pp)")
    print(f"    Delta LF               : {did['marginal_lf']:+.4f}  ({did['marginal_lf']*100:+.2f} pp)")
    print(f"    Expected additive gain : {did['expected_add']:+.4f}  ({did['expected_add']*100:+.2f} pp)")
    print(f"    Actual combined gain   : {did['actual_combined_gain']:+.4f}  ({did['actual_combined_gain']*100:+.2f} pp)")
    print(f"\n  Interaction (DiD):")
    sig_did = "YES (p<0.05)" if did["significant"] else "no"
    print(f"    Observed DiD  : {did['obs_did']:+.4f}  ({did['obs_did']*100:+.2f} pp)")
    print(f"    {did['ci_level']} CI      : [{did['ci_lo']:+.4f}, {did['ci_hi']:+.4f}]")
    print(f"    Bootstrap p   : {fmt_p(did['p_boot'])}  [{sig_did}]")
    if did["obs_did"] > 0.001:
        print("    -> Synergy: combined EXCEEDS the additive expectation")
    elif did["obs_did"] < -0.001:
        print("    -> Sub-additivity: combined FALLS SHORT of additive expectation")
    else:
        print("    -> Approximately additive")

    all_results["RQ3b"] = {"did": did}

    # GLMM (secondary)
    print("\n  [GLMM] Fitting linear mixed model on item-level data (secondary) ...")
    glmm = run_glmm(dfs)
    if "error" not in glmm:
        print(f"\n  GLMM Coefficients  (formula: {glmm['formula']}):")
        print(f"  {'Parameter':<25}  {'beta':>8}  {'p-value':>14}")
        print(f"  {'-'*55}")
        for param, coef in glmm["params"].items():
            pval     = glmm["pvalues"].get(param, float("nan"))
            pval_str = fmt_p(pval) if not math.isnan(pval) else "n/a"
            print(f"  {param:<25}  {coef:>+8.4f}  {pval_str:>14}")
        print(f"\n  NOTE: {glmm['note']}")
    else:
        print(f"  GLMM error: {glmm['error']}")
    all_results["RQ3b"]["glmm"] = glmm

    # =========================================================================
    # Summary table
    # =========================================================================
    section("Summary Table  [item-level McNemar, seed=0, N_boot=10000]")
    print(f"\n  Statistical unit: original matched TSV item IDs (no question-level collapse)")
    print(f"\n  {'RQ':<6}  {'Comparison':<34}  {'N items':>8}  {'Delta':>8}  "
          f"{'McNemar p':>14}  {'95% CI':>22}  {'Sig?':>5}")
    print("  " + "-" * 106)

    rows = [
        ("RQ1",  "Zero-shot -> Lexicon",   mc_rq1,    bci_rq1),
        ("RQ2",  "Zero-shot -> LF",        mc_rq2,    bci_rq2),
        ("RQ3a", "Lexicon -> Combined",    mc_3a_lex, bci_3a_lex),
        ("RQ3a", "LF -> Combined",         mc_3a_lf,  bci_3a_lf),
    ]
    for rq, lbl, mc, bci in rows:
        sig    = "YES" if mc["significant"] else "no"
        ci_str = f"[{bci['ci_lo']:+.4f}, {bci['ci_hi']:+.4f}]"
        print(f"  {rq:<6}  {lbl:<34}  {mc['n_items']:>8}  {mc['delta']:>+8.4f}  "
              f"{fmt_p(mc['p_value']):>14}  {ci_str:>22}  {sig:>5}")

    # DiD row
    sig    = "YES" if did["significant"] else "no"
    ci_str = f"[{did['ci_lo']:+.4f}, {did['ci_hi']:+.4f}]"
    print(f"  {'RQ3b':<6}  {'Lex x LF interaction (DiD)':<34}  {did['n_items']:>8}  "
          f"{did['obs_did']:>+8.4f}  {fmt_p(did['p_boot']):>14}  {ci_str:>22}  {sig:>5}")

    # ---- Save JSON ----------------------------------------------------------
    print("\n[5/5] Saving results ...")
    out_json = os.path.join(OUTPUT_DIR, f"statistical_results_{METRIC}.json")

    def _clean(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, pd.Series):   return obj.to_dict()
        if isinstance(obj, dict):        return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):        return [_clean(v) for v in obj]
        return obj

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(_clean(all_results), fh, indent=2, ensure_ascii=False)

    print(f"\n  Results saved -> {out_json}")
    print(f"\n{'=' * 70}\n  Done.\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
