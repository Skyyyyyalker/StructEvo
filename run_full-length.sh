#!/bin/bash
set -e

PROTEIN=$1             # supports: AAV, GFP
LEVEL=$2            # supports: medium, hard
TASK=$PROTEIN-$LEVEL
MAX_ROUND=15
SEED=42
CUDA=0
export CUDA_VISIBLE_DEVICES=$CUDA
echo Run task $TASK seed $SEED on cuda $CUDA

CANDIDATES_DIR=./candidates/$TASK/$SEED
PROXY_CKPT_DIR=./proxy/ckpts/$TASK/$SEED
TENSORBOARD_DIR=./tensorboard_logs/$TASK/$SEED
mkdir -p $CANDIDATES_DIR
mkdir -p $PROXY_CKPT_DIR
mkdir -p $TENSORBOARD_DIR

INITPOOL=./data/$PROTEIN/${LEVEL}_initpool.csv
STRUCT_FILE=./data/$PROTEIN/$PROTEIN.pdb
cp $INITPOOL $CANDIDATES_DIR/round_0.csv

for ((round_id=0; round_id<MAX_ROUND; round_id++)); do
    echo Training proxy for round $round_id 
    python3 -m structevo.reward.train \
        --task $TASK \
        --round_id $round_id \
        --seed $SEED \
        --num_epochs 30 \
        --structure_datafile $STRUCT_FILE \
        --train_datafile $CANDIDATES_DIR/round_$round_id.csv \
        --ckpt_load_path $PROXY_CKPT_DIR/round_$((round_id-1)).pth \
        --ckpt_save_path $PROXY_CKPT_DIR/round_$round_id.pth

    echo Running PPO for round $((round_id+1))
    python3 -m structevo.env_full \
        --task $TASK \
        --round $((round_id+1)) \
        --n_candidates 256 \
        --seed $SEED \
        --proxy_ckpt $PROXY_CKPT_DIR/round_$round_id.pth \
        --oracle_ckpt ./ckpts/ggs/$PROTEIN-oracle.ckpt \
        --init_sequence_path $CANDIDATES_DIR/round_$round_id.csv \
        --candidates_save_path $CANDIDATES_DIR/round_$((round_id+1)).csv \
        --structure_filepath $STRUCT_FILE \
        --tensorboard_logdir $TENSORBOARD_DIR \
        --tensorboard_logname $SEED

done