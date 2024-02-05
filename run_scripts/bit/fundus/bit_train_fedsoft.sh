#!/bin/bash

cd ../../../

trial=1

python main.py --config 'config.fundus.fedsoft' \
               --server 'bit' \
               --trial $trial \
               --run_name 'fedsoft_trial_'$trial \
               --run_notes 'trial '$trial': fedsoft' \
               --exp_name 'fl.fundus'