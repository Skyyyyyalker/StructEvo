#!/bin/bash

TASK=$1
LEVEL=$2

python3 -m structevo.evaluate_full --task $TASK-$LEVEL --final_round 15
