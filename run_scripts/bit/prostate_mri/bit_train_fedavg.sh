#!/bin/bash

cd ../../../

trial=0

python main.py --config 'config.prostate_mri.fedavg' \
               --server 'bit' \
               --trial $trial \
               --run_name 'fedavg_trial_'$trial \
               --run_notes 'trial '$trial': fedavg' \
               --exp_name 'fl.prostate_mri'