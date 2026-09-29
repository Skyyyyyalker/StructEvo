"""
Calculate mean/median fitness、diversity、mean/median novelty metrics 
on last round top-128 candidates for full-length tasks
"""
import argparse
import os
import pandas as pd
from tqdm import tqdm
from structevo.utils import get_logger, EvalRunner
log = get_logger("Evaluate")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['GFP-medium', 'GFP-hard', 'AAV-medium', 'AAV-hard'])
    parser.add_argument('--final_round', type=int, default=15, help="which round candidates to evaluated")
    parser.add_argument('--topk', type=int, default=128, help='top k candidates used to evaluated')
    parser.add_argument('--use_normalization', action='store_false', help='default: True')
    return parser.parse_args()

args = get_args()
protein, level = args.task.split('-')
initpool_csv = f"./data/{protein}/{level}_initpool.csv"
gt_csv = f"./data/{protein}/ground_truth.csv"
candidates_dir = f"./candidates/{args.task}/"
savepath = os.path.join("./results", f"{args.task}_round{args.final_round}.csv")
eval_runner = EvalRunner(gt_csv, initpool_csv, use_normalization=args.use_normalization)

testnum_range = sorted([int(i) for i in os.listdir(candidates_dir)])
log.info(f"Collected {len(testnum_range)} testnums for our results: {testnum_range}")
all_results = []
for testnum in tqdm(testnum_range, desc=f"Processing {len(testnum_range)} testnums"):
    candidates_file = os.path.join(candidates_dir, str(testnum), f"round_{args.final_round}.csv")
    prev_files = [os.path.join(candidates_dir, str(testnum), f"round_{i}.csv") for i in range(1, args.final_round+1)]
    df_all = pd.concat([pd.read_csv(f) for f in prev_files]).reset_index(drop=True)
    df_topk = df_all.sort_values(by="GroundTruth", ascending=False).iloc[:args.topk]
    df_results = eval_runner.evaluate_sequences(df_topk)
    result = pd.DataFrame({
        'mean_fitness': [df_results["mean_fitness"][0]],
        'max_fitness': [df_results["max_fitness"][0]],
        'diversity': [df_results["mean_diversity"][0]],
        'novelty': [df_results["mean_novelty"][0]],
        'source_path': [candidates_file]
    })
    all_results.append(result)

all_results = pd.concat(all_results).reset_index(drop=True)
df_summary = pd.DataFrame({
    "mean_fitness": [all_results['mean_fitness'].mean()], 
    "max_fitness": [all_results['max_fitness'].mean()],
    "diversity": [all_results['diversity'].mean()],
    "novelty": [all_results['novelty'].mean()],
    'std_mean_fitness': [all_results['mean_fitness'].std()],
    'std_max_fitness': [all_results['max_fitness'].std()],
    'std_diversity': [all_results['diversity'].std()],
    'std_novelty': [all_results['novelty'].std()]
})
# Save results csv
os.makedirs(os.path.dirname(savepath), exist_ok=True)
df_summary.to_csv(savepath, index=False, float_format='%.4f')
log.info(f'Final round results saved at {savepath}')
