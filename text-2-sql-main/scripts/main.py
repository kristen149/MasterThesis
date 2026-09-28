from dataset_analysis import execute_analysis
import pandas as pd
from results_analysis import process_results_analyis
from results_grouped_by_utterance import process_results_grouped_by_utterance

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=str)
    parser.add_argument("--metric", type=str, default='EM')
    parser.add_argument("--perspective", type=str, default='process_mining')
    parser.add_argument("--result_folder", type=str, default='openAI-representation_0-shot')

    args = parser.parse_args()

    if args.execution == 'dataset_statistics':
      execute_analysis(args.perspective)
    elif args.execution == 'result_analysis':
      process_results_analyis(args.perspective, args.metric, args.result_folder)
    elif args.execution == 'results_grouped_by_utterance':
      process_results_grouped_by_utterance()
    else:
      print('Inform a execution to process')
    