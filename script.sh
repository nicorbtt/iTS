#!/bin/zsh

# python3 src/main.py -s --dataset_name M5 --model deepAR --distribution_head poisson &
# python3 src/main.py -s --dataset_name M5 --model deepAR --distribution_head negbin &
# python3 src/main.py -s --dataset_name M5 --model deepAR --distribution_head tweedie &
# python3 src/main.py -s --dataset_name M5 --model deepAR --distribution_head poisson --scaling mean-demand &
# python3 src/main.py -s --dataset_name M5 --model deepAR --distribution_head negbin --scaling mean-demand &
# python3 src/main.py -s --dataset_name M5 --model deepAR --distribution_head tweedie --scaling mean-demand &
# python3 src/main.py -s --dataset_name M5 --model transformer --distribution_head poisson &
# python3 src/main.py -s --dataset_name M5 --model transformer --distribution_head negbin &
# python3 src/main.py -s --dataset_name M5 --model transformer --distribution_head tweedie &
python3 src/main.py -s --dataset_name M5 --model transformer --distribution_head poisson --scaling mean-demand &
python3 src/main.py -s --dataset_name M5 --model transformer --distribution_head negbin --scaling mean-demand &
python3 src/main.py -s --dataset_name M5 --model transformer --distribution_head tweedie --scaling mean-demand &