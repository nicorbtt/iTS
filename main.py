# generate for me a python script that given the model, launches global.py with the rigght parameters

import subprocess
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iLocGlob: global vs local models for intermittent time series forecasting")
    parser.add_argument('--dataset_name', type=str, choices=['OnlineRetail', 'Auto', 'RAF', 'carparts', 'syph', 'M5', 'crime', 'VN1', 'UCI'], required=True, help='Specify dataset name')
    parser.add_argument('--model', type=str, choices=['deepAR','transformer','informer', 'autoformer', 'patchTST', 'tide', 'feedforward', 'dlinear', "ISQ", "iETS", "tweedieGP", "MW"], required=True, help="Specify model")
    parser.add_argument('--distribution_head', type=str, choices=['poisson','negbin', 'tweedie', 'zinb'], default=None, help="Specify distribution_head, default is None")
    parser_args = parser.parse_args()
    
    if parser_args.model in ['deepAR','transformer','informer', 'autoformer', 'patchTST', 'tide', 'feedforward', 'dlinear']:
        if parser_args.distribution_head is None:
            raise ValueError("distribution_head must be specified for the selected model.")
        cmd = [
            sys.executable,  # Use the current Python interpreter
            os.path.join('src', 'global.py'),
            '--dataset_name', parser_args.dataset_name,
            '--model', parser_args.model,
            '--distribution_head', parser_args.distribution_head
        ]

    if parser_args.model in ['ISQ', 'iETS', 'tweedieGP', 'MW']:
        cmd = [
            sys.executable,  # Use the current Python interpreter
            os.path.join('src', 'local.py'),
            '--dataset_name', parser_args.dataset_name,
            '--model', parser_args.model
        ]
        
    subprocess.run(cmd)