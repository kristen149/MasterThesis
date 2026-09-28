import pandas as pd
import os
import glob

DATASET_PATH = '../data/dataset/text2sql4pm.tsv'
RESULTS_BASE_PATH = '../results/english/'

class LoaderResults(object):
    def __init__(self, *args, **kwargs):
        self.results = {}

    def get_data_frame_results(self, path, model):
        tmp_df = pd.read_csv(path, sep='\t', header=None)
        tmp_df.columns = ['id_' + model, 'predicted_' + model, 'gold_' + model, 'hardness_' + model, 'score_' + model]
        return tmp_df

    def load_results(self, models_r: dict):
        for key, value in models_r.items():
            file_path = value['path'] + 'scores_' + key + '_' + value['metric'] + '.tsv'
            tmp_df = self.get_data_frame_results(file_path, key)
            self.results[key] = tmp_df

    def concat_all_results(self):
        tmp_df = pd.DataFrame()
        for value in self.results.values():
            tmp_df = pd.concat([tmp_df, value], axis=1)
        return tmp_df

def load_dataset():
    dataset = pd.read_csv(DATASET_PATH, sep='\t')
    dataset = dataset.dropna(how='all')
    return dataset

def detect_model_key(result_folder, metric):
    eval_path = os.path.join(RESULTS_BASE_PATH, result_folder, 'evaluations')
    matches = glob.glob(os.path.join(eval_path, f'scores_*_{metric}.tsv'))
    if not matches:
        raise FileNotFoundError(f"No TSV file found for metric '{metric}' in {eval_path}")
    filename = os.path.basename(matches[0])
    model_key = filename[len('scores_'):filename.rfind(f'_{metric}.tsv')]
    return model_key

def load_results(metric, result_folder='openAI-representation_0-shot'):
    dataset = load_dataset()
    model_key = detect_model_key(result_folder, metric)
    eval_path = os.path.join(RESULTS_BASE_PATH, result_folder, 'evaluations') + '/'
    results_en = LoaderResults()
    results_en.load_results({model_key: {'path': eval_path, 'metric': metric}})
    shot0_en = results_en.concat_all_results()
    return dataset, shot0_en, model_key