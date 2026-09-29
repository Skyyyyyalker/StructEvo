import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from structevo.utils import get_logger
log = get_logger("Evaluate")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['GB1', 'PhoQ'])
    parser.add_argument('--final_round', type=int, default=3, help="which round candidates to evaluated")
    parser.add_argument('--topk', type=int, default=96, help='top k candidates used to evaluated')
    return parser.parse_args()

args = get_args()

savepath = f"./results/{args.task}_round{args.final_round}.csv"
candidates_dir = f"./candidates/{args.task}"
proxy_prediction_dir = f"./proxy/ckpts/{args.task}"
gt_path = f"./data/{args.task}/ground_truth.csv"
df_gt = pd.read_csv(gt_path).reset_index(drop=True)
combo2gt = {df_gt["AACombo"][i]: df_gt["GroundTruth"][i] for i in range(len(df_gt))}

testnum_list = []
for x in os.listdir(candidates_dir):
    if x == "clade_init": continue
    testnum_list.append(int(x))
testnum_list = sorted(testnum_list)
log.info(f"Found {len(testnum_list)} testnums for {args.task}: {testnum_list}")

MAX_FITS = {
    "GB1": 8.761966,
    "PhoQ": 133.594270,
}
MAX_FIT = MAX_FITS[args.task]
MIN_FIT = 0.0

def maxmin_normalize(fit):
    return (fit - MIN_FIT) / (MAX_FIT - MIN_FIT)

def get_groundtruth(combo_list):
    gt_list = []
    for combo in combo_list:
        gt_value = combo2gt[combo] if combo in combo2gt else 0.0
        gt_list.append(gt_value)    
    return gt_list

all_results = []
for testnum in tqdm(testnum_list):
    candidates_file = os.path.join(candidates_dir, f"{testnum}/round_{args.final_round}.csv")
    df_trainset = pd.read_csv(candidates_file)
    train_combos = set(df_trainset["AACombo"].tolist())
    # top 96 by prediction
    prediction_file = os.path.join(proxy_prediction_dir, f"{testnum}/round_{args.final_round}.csv")
    df_reward = pd.read_csv(prediction_file)
    df_reward = df_reward.sort_values(by="PredictedFitness", ascending=False).reset_index(drop=True)
    df_reward["GroundTruth"] = get_groundtruth(df_reward["AACombo"].tolist())
    df_reward_top96 = df_reward[~df_reward["AACombo"].isin(train_combos)].head(96)
    
    df_combined = pd.concat([df_trainset, df_reward_top96]).drop(columns=["Unnamed: 0"])
    df_combined = df_combined.sort_values(by="GroundTruth", ascending=False).reset_index(drop=True)
    max_fitness = df_combined["GroundTruth"][0]
    mean_fitness = np.mean(df_combined["GroundTruth"][:args.topk])
    ndcg = ndcg_score(
        np.expand_dims(np.array(df_reward['GroundTruth']), axis=0),
        np.expand_dims(np.array(df_reward['PredictedFitness']), axis=0)
    )

    result = pd.DataFrame({
        'max_fitness': [maxmin_normalize(max_fitness)],
        'mean_fitness': [maxmin_normalize(mean_fitness)],
        'ndcg': [ndcg]
    })
    all_results.append(result)

all_results = pd.concat(all_results).reset_index(drop=True)
df_summary = pd.DataFrame({
    "max_fitness": [all_results['max_fitness'].mean()],
    "mean_fitness": [all_results['mean_fitness'].mean()], 
    "ndcg": [all_results['ndcg'].mean()],
    'std_mean_fitness': [all_results['mean_fitness'].std()],
    'std_max_fitness': [all_results['max_fitness'].std()],
    'std_ndcg': [all_results['ndcg'].std()]
})
# Save results csv
os.makedirs(os.path.dirname(savepath), exist_ok=True)
all_results.to_csv(savepath.replace('.csv', '_all_results.csv'), index=False, float_format='%.4f')
df_summary.to_csv(savepath, index=False, float_format='%.4f')
log.info(f'Results saved at {savepath}')
