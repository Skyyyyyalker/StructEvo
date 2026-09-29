#!/bin/bash
set -e

PROTEIN=$1             # supports: GB1, PhoQ
TASK=$PROTEIN
MAX_ROUND=3
SEED=2
CUDA=0
export CUDA_VISIBLE_DEVICES=$CUDA
echo Run task $TASK seed $SEED on cuda $CUDA

CANDIDATES_DIR=./candidates/$TASK/$SEED
PROXY_CKPT_DIR=./proxy/ckpts/$TASK/$SEED
TENSORBOARD_DIR=./tensorboard_logs/$TASK/$SEED
STRUCT_FILE=./data/$PROTEIN/$PROTEIN.pdb
mkdir -p $CANDIDATES_DIR
mkdir -p $PROXY_CKPT_DIR
mkdir -p $TENSORBOARD_DIR

# TODO: RUN CLADE HERE and put initial file at $CANDIDATES_DIR/round_0.csv
# we provide example files at clade_init dir if you meet trouble in running CLADE.
cp ./candidates/$TASK/clade_init/seed_$SEED.csv $CANDIDATES_DIR/round_0.csv

for ((round_id=0; round_id<MAX_ROUND; round_id++)); do
    echo Training proxy for round $round_id 
    python3 -m structevo.reward.train \
        --task $TASK \
        --round_id $round_id \
        --seed $SEED \
        --num_epochs 50 \
        --structure_datafile $STRUCT_FILE \
        --train_datafile $CANDIDATES_DIR/round_$round_id.csv \
        --ckpt_save_path $PROXY_CKPT_DIR/round_$round_id.csv \
        --predict_space_path ./data/$PROTEIN/all_seq.csv

    echo Running PPO for round $((round_id+1))
    python3 -m structevo.env_4site \
        --task $TASK \
        --round $((round_id+1)) \
        --n_candidates 96 \
        --seed $SEED \
        --proxy_ckpt $PROXY_CKPT_DIR/round_$round_id.csv \
        --ground_truth_path ./data/$PROTEIN/ground_truth.csv \
        --init_sequence_path $CANDIDATES_DIR/round_$round_id.csv \
        --candidates_save_path $CANDIDATES_DIR/round_$((round_id+1)).csv \
        --structure_filepath $STRUCT_FILE \
        --tensorboard_logdir $TENSORBOARD_DIR \
        --tensorboard_logname $SEED
done

# Prediction for round 3
echo Training proxy for round 3
python3 -m structevo.reward.train \
    --task $TASK \
    --round_id 3 \
    --seed $SEED \
    --num_epochs 50 \
    --structure_datafile $STRUCT_FILE \
    --train_datafile $CANDIDATES_DIR/round_3.csv \
    --ckpt_save_path $PROXY_CKPT_DIR/round_3.csv \
    --predict_space_path ./data/$PROTEIN/all_seq.csv
