# GeneSis: A Generative Approach to Substitutes in Context

This repository contains the instructions to reproduce the experiments in the GeneSis paper.

## 1. Setup a new environment

a) Create and activate a new conda environment with 

```conda create -n genesis python=3.8```\
```conda activate genesis```

b) Install pythorch following the instructions at https://pytorch.org/ 

c) Install the requirements with 

```pip install -r requirements.txt```

## 2. Download additional data and checkpoints

If you want to experiment with the datasets generated from SemCor, download the generated_datasets.tar.gz from https://tinyurl.com/ym7mebv9 and put the files under the ```data/``` directory. 
Note that the name of each file has the following format: ```semcor_{similarity_threshold}_{split_size}_train.tsv```. The dataset without ```{split_size}``` is the whole dataset.
If you want to test one of the models described in the paper, download the checkpoints from https://tinyurl.com/jnc6rk44 and move the external folder under the ```checkpoints/``` directory.
The structure of each checkpoint folder is the following:
``` 
|--- bart_{seed}_pt_{training_dataset}_drop_{dropout}_enc_lyd_{encoder_layerdropout}_dec_lyd_{decoder_layerdropout} \
     |--- beams_15_return_3 
          |--- checkpoints 
          |    |--- best_checkpoint.ckpt
          |--- input_training_dataset.tsv # dataset used at training time

```

## 3. Train 

Setup the config file ```config/train.yml``` following to the comments. \
Train the model with ```PYTHONPATH=$(pwd) python src/train.py --config_path config/train.yml```. \
You can optionally pass the following parameters:
```
--dropout # a new value for dropout, will override the one defined in config/train.yml
--encoder_layerdropout # a new value for encoder layer dropout, will override the on in config/train.yml
--decoder_layerdropout # a new value for decoder layer dropout, will override the on in config/train.yml
--seed # value for the seed. The default one is 0.
--ckpt # path to a checkpoint of a pre-trained model. It can be given as a parameter in order to continue training on a different dataset, defined in the 'finetune' field of the config/train.yml

```
The checkpoints for each run will be saved in the folder ```checkpoints/ bart_{seed}_pt_{training_dataset}_drop_{dropout}_enc_lyd_{encoder_layerdropout}_dec_lyd_{decoder_layerdropout}/beams_{beam_size}_return_{return_sequences}/checkpoints/
