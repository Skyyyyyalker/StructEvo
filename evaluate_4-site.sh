#!/bin/bash

TASK=$1

python3 -m structevo.evaluate_4site --task $TASK --final_round 3
