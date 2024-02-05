# Federated semi-supervised learning with marginal feature distribution consistency for medical image segmentation

## Introduction

This repo contains the code and configuration files to reproduce results of federated semi-supervised learning with marginal feature distribution consistency for medical image segmentation.

## Environments

Please refer to ```requirements.txt``` for the version of python libraries I used. I don't suggest to directlly install
all by ```pip install -r requirements.txt``` because they are many unnecessary dependencies.


## Datasets 

* Prostate MRI Segmentation Dataset (ProstateMRI) [[URL](https://liuquande.github.io/SAML/)]
* Optic Disc/Cup Segmentation Dataset (REFUGE) [[URL](https://refuge.grand-challenge.org/Download/)]

## Training
### For Prostate MRI Segmentation Dataset (ProstateMRI)
```
cd FedMFDC/run_scripts/bit/prostate_mri
sh bit_train_fedsoft.sh 
```
### For Optic Disc/Cup Segmentation Dataset (REFUGE)
```
cd FedMFDC/run_scripts/bit/fundus
sh bit_train_fedsoft.sh 
```

