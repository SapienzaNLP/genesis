# GeneSis: A Generative Approach to Substitutes in Context

This repository contains the instructions to reproduce the experiments in the [GeneSis paper](https://www.researchgate.net/publication/355646366_GeneSis_A_Generative_Approach_to_Substitutes_in_Context), accepted at EMNLP 2021.
When using this work, please cite it as follows:

```
@inproceedings{lacerraetal:2021,
  title={ Gene{S}is: {A} {G}enerative {A}pproach to {S}ubstitutes in {C}ontext},
  author={Lacerra, Caterina and Tripodi, Rocco and Navigli, Roberto},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  publisher={Association for Computational Linguistics},
  year={2021},
  address={Punta Cana, Domenican Republic}
 }
```

## 1. :gear: Setup a new environment 

a) Create and activate a new conda environment with 

```conda create -n genesis python=3.8```\
```conda activate genesis```

b) Install pythorch following the instructions at https://pytorch.org/ 

c) Install the requirements with 

```pip install -r requirements.txt```

## 2. :shopping_cart: Download additional data and checkpoints

If you want to experiment with the datasets generated from SemCor, download the [generated datasets](https://drive.google.com/uc?export=download&id=1keUU1zjriXCi3nZmIsePNx02i-Bt87dX) and put the files under the ```data/``` directory.
Note that the name of each file has the following format: ```semcor_{similarity_threshold}_{split_size}_train.tsv```. The dataset without ```{split_size}``` is the whole dataset.
If you want to test one of the models described in the paper, download the [checkpoint](https://drive.google.com/uc?export=download&id=12G--HAMSPadoxj_K8nD_GZ8oXaUs85o-) and move the external folder under the ```output/``` directory.
The structure of each output subfolder is the following:
``` 
|--- bart_{seed}_pt_{training_dataset}_drop_{dropout}_enc_lyd_{encoder_layerdropout}_dec_lyd_{decoder_layerdropout} \
     |--- beams_15_return_3 
          |--- checkpoints 
          |    |--- best_checkpoint.ckpt
          |--- input_training_dataset.tsv # dataset used at training time

```

## 3. :train: Train 

Setup the config file ```config/train.yml``` following to the comments. \
Train the model with ```PYTHONPATH=$(pwd) python src/train.py --config_path config/train.yml```. \
You can optionally pass the following parameters:
``` bash
--dropout # a new value for dropout, will override the one defined in config/train.yml
--encoder_layerdropout # a new value for encoder layer dropout, will override the on in config/train.yml
--decoder_layerdropout # a new value for decoder layer dropout, will override the on in config/train.yml
--seed # value for the seed. The default one is 0.
--ckpt # path to a checkpoint of a pre-trained model. It can be given as a parameter in order to continue training on a different dataset, defined in the 'finetune' field of the config/train.yml

```
The checkpoints for each run will be saved in the folder 
```output/ bart_{seed}_pt_{training_dataset}_drop_{dropout}_enc_lyd_{encoder_layerdropout}_dec_lyd_{decoder_layerdropout}/beams_{beam_size}_return_{return_sequences}/checkpoints/```

## 4. :test_tube: Test

To test a trained model on the lexical substitution task, run
```PYTHONPATH=$(pwd) python src/test.py --config_path config/train.yml --ckpt path_to_checkpoint --cvp vocab/wordnet_vocab --cut_vocab ```

There are several parameters that can be defined:
``` bash 
--ckpt # path to the checkpoint to test. It is REQUIRED.
--suffix # suffix string for output file names. It is REQUIRED.
--cvp # path to the vocabulary to use. It is a REQUIRED parameter (give '' for testing without cutting on vocab)
--test # flag to test on test. If it is not given as a parameter, the model will be EVALUATED ON THE DEV SET!
--beams # beam size 
--sequences # number of returned beams 
--cut_vocab # flag to cut the output on a reduced vocabulary
--backoff # flag to use the backoff strategy
--baseline # flag to compute the baseline
--embedder # name of HuggingFace model to be used as embedder for computing contextualized representations used for ranking
--cuda_device # GPU id. If it's not given, the model will be tested on CPU
```

## 5. :clipboard: Output Files

The test script will produce several output files in the ```output/ bart_{seed}_pt_{training_dataset}_drop_{dropout}_enc_lyd_{encoder_layerdropout}_dec_lyd_{decoder_layerdropout}/beams_{beam_size}_return_{return_sequences}/output_files/``` folder. 
The most important one is named ```output_{suffix}_{test_dataset_name}.txt``` and contains the raw (without cut on the datest, without backoff strategy), formatted, for each instance, as follows:

```bash
target_word.POS instance_id [target_indexes] # ex: rest.NOUN 1922 [1]
target_context                               # ex: the rest is up to you .
gold_substitutes                             # ex: #gold: remainder: 5.0 balance: 1.0

sequence 1                                   # ex: remainder, remainder of the work, the rest of it, balance, extra,
sequence 2                                   ...
sequence returned_sequences                  # ex: remainder, remainder of the work, the rest of it, the balance, extra
```
This file is used to clean the output and postprocess it before feeding it to the scorer, called by ```test.py```. 
Thus, if you have already computed this file we can directly evalute the trained model (for example, trying with or without the output vocabulary) without testing it again, as described in the next section.
Two other output files are ```{test_dataset}_cut_per_target_{suffix}_best.txt``` and ```{test_dataset}_cut_per_target_{suffix}_oot.txt```, that format the output as required from the Perl scorer for the task evaluation, and finally ```{test_dataset}_cut_per_target_hr_{suffix}_output.txt``` that contains a more readable version of the model output, formatted as follows:

```bash
target_word.POS instance_id [target_indexes] # ex: rest.NOUN 1922 [1]
target_context                               # ex: the rest is up to you .
gold_substitutes                             # ex: #gold: remainder balance
collected_generations                        # ex: #generated: remainder, balance, extra, other, the, all
clean_output                                 # ex: #clean: remainder: 0.88, balance: 0.75, rest: 1.0, whole_rest: 0.83, remnant: 0.79 ...
```
The ```clean_output``` row contains the output of the model after the vocab cut (if ```--cut_vocab``` is specified) and with fallback strategy (if ```--backoff```). The floats are the cosine similarities between target and substitute.

## 6. :microscope: Task Evaluation

It is possible to try different configurations of a model already tested by running ```PYTHONPATH=$(pwd) python src/task_evaluation.py --config_path config/train.yml```. The parameters are the same of ```test.py```, plus ```--finetune``` that has to be set if you trained your model with pre-training + finetuning. Note that the only parameters that can be changed without re-testing the model are ```--cvp```, ```--cut_vocab``` and ```--backoff```.

# :spiral_notepad: Dataset Generation
In order to generate new silver data, is it possible to use the scripts in the ```src/dataset_generation``` folder.

a) ```annotate_semcor.py``` builds an input file properly formatted for the model, starting from a pair of files in the Raganato [framework](http://lcl.uniroma1.it/wsdeval/)'s format

b) The generated file is the input required by ```generate_dataset.py```, to be provided through the ```--sentences_path``` parameter. In this case, the config file to give as parameter is ```config/dataset_generation.yml```. All the other parameters have been discussed in the sections above.

c) ```dataset_cleaning.py``` cleans out the generated dataset. The ```--threshold``` parameter is the threshold on cosine similarity between target and substitutes.

d) Finally, ```semcor_splits.py``` produces a split of the clean dataset. The parameters are commented in the code.


# Acknowledgments

The authors gratefully acknowledge the support of the ERC Consolidator Grant MOUSSE No. 726487 under the European Union’s Horizon 2020 research and innovation programme.

This work was supported in part by the MIUR under grant “Dipartimenti di eccellenza 2018-2022” of the Department of Computer Science of the Sapienza University of Rome.

# License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/)