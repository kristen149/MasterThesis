import os, glob, csv, math
import pandas as pd
import numpy as np

dataset_path = 'data/dataset/text2sql4pm.tsv'
dataset = pd.read_csv(dataset_path, sep='\t', quoting=csv.QUOTE_NONE).dropna(how='all').reset_index(drop=True)

results_dir = 'results/english'
runs = {
    'zeroshot': 'oss_qwopus_glm_18b_healed_q4_k_m_zeroshot',
    'lexicon': 'oss_qwopus_glm_18b_healed_q4_k_m_lexicon',
    'lf': 'oss_qwopus_glm_18b_healed_q4_k_m_lf',
    'lexicon-lf': 'oss_qwopus_glm_18b_healed_q4_k_m_lexicon-lf'
}

data_frames = {}

print("=== 1. DESCRIPTIVE PARAPHRASE METRICS ===\n")

for label, folder in runs.items():
    eval_dir = os.path.join(results_dir, folder, 'evaluations')
    ex_file = glob.glob(os.path.join(eval_dir, 'scores_*_EX.tsv'))[0]
    em_file = glob.glob(os.path.join(eval_dir, 'scores_*_EM.tsv'))[0]
    
    df_ex = pd.read_csv(ex_file, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    df_em = pd.read_csv(em_file, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    
    df = dataset.copy()
    df['score_ex'] = df_ex[4].values
    df['score_em'] = df_em[4].values
    data_frames[label] = df
    
    base_df = df[df['Base_paraphrase'] == 'base']
    para_df = df[df['Base_paraphrase'] == 'paraphrase']
    
    b_ex_cnt, b_ex_tot = base_df['score_ex'].sum(), len(base_df)
    b_ex_pct = (b_ex_cnt / b_ex_tot) * 100
    
    p_ex_cnt, p_ex_tot = para_df['score_ex'].sum(), len(para_df)
    p_ex_pct = (p_ex_cnt / p_ex_tot) * 100
    
    o_ex_cnt, o_ex_tot = df['score_ex'].sum(), len(df)
    o_ex_pct = (o_ex_cnt / o_ex_tot) * 100
    
    ex_delta = p_ex_pct - b_ex_pct
    
    print(f"[{label:<10s}] Base (n={b_ex_tot}): {b_ex_pct:5.2f}% ({b_ex_cnt}/{b_ex_tot}) | Para (n={p_ex_tot}): {p_ex_pct:5.2f}% ({p_ex_cnt}/{p_ex_tot}) | Overall: {o_ex_pct:5.2f}% ({o_ex_cnt}/{o_ex_tot}) | Delta: {ex_delta:+6.2f}%")

print("\n=== 2. 10,000-ITERATION PAIRED GROUP BOOTSTRAP TEST FOR ROBUSTNESS ===\n")

groups = dataset['Group_id'].unique() # 205 canonical question groups
np.random.seed(42)
n_boot = 10000

# Pre-index for fast execution
group_indices = {g: dataset[dataset['Group_id'] == g].index.values for g in groups}
group_base_mask = {g: dataset.loc[dataset['Group_id'] == g, 'Base_paraphrase'].values == 'base' for g in groups}
group_para_mask = {g: dataset.loc[dataset['Group_id'] == g, 'Base_paraphrase'].values == 'paraphrase' for g in groups}

diff_degradation_comb_vs_lf = []

for _ in range(n_boot):
    # Resample 205 question groups with replacement
    boot_groups = np.random.choice(groups, size=len(groups), replace=True)
    
    lf_b_sums, lf_b_cnts = 0, 0
    lf_p_sums, lf_p_cnts = 0, 0
    comb_b_sums, comb_b_cnts = 0, 0
    comb_p_sums, comb_p_cnts = 0, 0
    
    for g in boot_groups:
        idx = group_indices[g]
        b_m = group_base_mask[g]
        p_m = group_para_mask[g]
        
        lf_scores = data_frames['lf']['score_ex'].values[idx]
        comb_scores = data_frames['lexicon-lf']['score_ex'].values[idx]
        
        lf_b_sums += np.sum(lf_scores[b_m])
        lf_b_cnts += np.sum(b_m)
        lf_p_sums += np.sum(lf_scores[p_m])
        lf_p_cnts += np.sum(p_m)
        
        comb_b_sums += np.sum(comb_scores[b_m])
        comb_b_cnts += np.sum(b_m)
        comb_p_sums += np.sum(comb_scores[p_m])
        comb_p_cnts += np.sum(p_m)
        
    deg_lf = (lf_p_sums / lf_p_cnts * 100) - (lf_b_sums / lf_b_cnts * 100)
    deg_comb = (comb_p_sums / comb_p_cnts * 100) - (comb_b_sums / comb_b_cnts * 100)
    
    # Store degradation reduction: deg_comb - deg_lf (e.g. -4.58% - (-10.57%) = +5.99% points)
    diff_degradation_comb_vs_lf.append(deg_comb - deg_lf)

diff_arr = np.array(diff_degradation_comb_vs_lf)
mean_diff = np.mean(diff_arr)
ci_low, ci_high = np.percentile(diff_arr, [2.5, 97.5])
p_boot = np.mean(diff_arr <= 0)

print(f"Iterations: {n_boot}")
print(f"Degradation Reduction (lexicon-lf vs lf): +{mean_diff:.2f}% points")
print(f"95% Non-parametric Bootstrap CI: [{ci_low:+.2f}%, {ci_high:+.2f}%]")
print(f"Empirical Bootstrap p-value: p = {p_boot:.4f} ({'Statistically Significant' if p_boot < 0.05 else 'Not Significant'})")
