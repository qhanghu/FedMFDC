#!/bin/bash

cd ../../../

trial=0

python main.py --config 'config.prostate_mri.fedprox' \
               --server 'bit' \
               --trial $trial \
               --run_name 'fedprox_trial_'$trial \
               --run_notes 'trial '$trial': fedprox' \
               --exp_name 'fl.prostate_mri'