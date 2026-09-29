import pandas as pd
from tqdm import tqdm
import argparse
import os
from structevo.utils import get_logger

log = get_logger("process_clade")

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default="../data/GB1.csv")
parser.add_argument('--old_path', type=str, default="xx/InputValidationData.csv")
parser.add_argument('--final_path', type=str, default="xx/round_0.csv")
args = parser.parse_args()

if os.path.exists(args.final_path):
    log.info("Processed CLADE candidates is already finished. Skipped.")
    exit(0)

if not os.path.exists(os.path.dirname(args.final_path)):
    log.info(f"Mkdir: {os.path.dirname(args.final_path)}")
    os.makedirs(os.path.dirname(args.final_path))

df_gt = pd.read_csv(args.gt_path)
df_old = pd.read_csv(args.old_path)
seqs = []
gts = []
for i in tqdm(range(len(df_old))):
    combo = df_old['AACombo'][i]
    gt = df_gt[df_gt['AACombo']==combo].iloc[0]["GroundTruth"]
    seq = df_gt[df_gt['AACombo']==combo].iloc[0]["Sequence"]
    gts.append(gt)
    seqs.append(seq)
df_old['Sequence'] = seqs
df_old['GroundTruth'] = gts
df_old = df_old.drop(columns=['Fitness', 'Cluster'])
df_old.to_csv(args.final_path)
log.info(f"Save candidates to {args.final_path}")
