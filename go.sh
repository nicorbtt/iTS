#!/bin/zsh

DATASET_NAME="M5"

run_negbin() {
    local seed=$1
    python3 src/main.py -log --dataset_name $DATASET_NAME --model transformer --distribution_head negbin --scaling mean-demand --seed $seed &
    #python3 src/main.py -log --dataset_name $DATASET_NAME --model deepAR --distribution_head negbin --scaling mean-demand --seed $seed &
}

run_tweedie() {
    local seed=$1
    python3 src/main.py -log --dataset_name $DATASET_NAME --model transformer --distribution_head tweedie --scaling mean-demand --seed $seed &
    #python3 src/main.py -log --dataset_name $DATASET_NAME --model deepAR --distribution_head tweedie --scaling mean-demand --seed $seed &
}

run_tweediefix() {
    local seed=$1
    python3 src/main.py -log --dataset_name $DATASET_NAME --model transformer --distribution_head tweedie-fix --scaling mean-demand --seed $seed &
    #python3 src/main.py -log --dataset_name $DATASET_NAME --model deepAR --distribution_head tweedie-fix --scaling mean-demand --seed $seed &
}

for seed in {2..5}; do
    run_negbin $seed
    wait
    run_tweedie $seed
    wait
    run_tweediefix $seed
    wait
done
