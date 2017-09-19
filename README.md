# Memory Network

This project is an implementation of the Memory Network (see [PAPER](https://arxiv.org/abs/1410.3916)).

People : David Panou

Organization : UPMC - Master Data Science

- :white_check_mark: Embeddings (inputs & memory )
- :white_check_mark: Relational Inference
- :white_medium_square: Results and Model Analysis
- :white_medium_square: Handling other entries than OneHot (Look-up Table for pretrained embeddings)
- :white_medium_square: Adding Optim package for better sgd (momentum and others)

## Memory Network

---

### Create the dataset

`python preprocessing/preprocessing.py`

You can adjust the level of detail in the generated answer using the `level` argument.  (default is `-level  0`) `-level 1` compute sentence instead of words.

### transform csv data into torch tensor (only if recomputed csv)

`th main.lua -generate_dataset 1`

### Perform Memory Network Training

`th main.lua -train_mem_net 1`  To run with default parameters

### Go into further parametring 

`th main.lua -train_mem_net 1 -num_mem 10 -feature_dim 174 -voc_size 58`

The written parameters above are the default one.

Be careful that :
    - voc_size is >= to the number of word in the dataset

---

The different modules can be found under `models/*_module.lua`

The current `main.lua` replicates the Basic Question-Answering task described in the paper, but the implementation should be able to handle other tasks.

Results are stored in `results`

## Results

Upcoming