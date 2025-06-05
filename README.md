# Neuro EWC on LLM

This is the implementation code for the preprint [Full-Parameter Continual Pretraining of Gemma2: Insights into Fluency and Domain Knowledge](https://arxiv.org/abs/2505.05946).

This docker has several files with the suffix `_example`.
Inspect them, modify as necessary and save as files without the suffix.

## Installation

Modify `run_docker_example.sh` with any additional volumes where data is stored
by inserting the following lines:

`-v <VOLUME_LOCAL>:<VOLUME_DOCKER> \`

Save it as `run_docker.sh`.
Then modify `.env_example` with the desired environment variables.
Since the docker runs as a user, and not root,
I suggest setting the `HOME` variable.
Rename it to `.env`.
Then build the image and run container:
```bash
docker build -t llm_ewc .
bash run_docker.sh
```

## Running

### Computing fisher matrix

We created a handy script that you need to modify at `compute_ewc_params_example.sh`.
You must set the locations for where the base model is, 
where the EWC params should be saved
and where the cache should be located.

Remove the suffix, and run using the following:
```bash
bash compute_ewc_params.sh
```

### Training

Then, after editing paths in the file `run_gemma2-2b_culturaX_example.sh` and removing the suffix, the training can be run:

```bash
bash run_gemma2-2b_culturaX.sh 1e2
```

Or several of them can be run:

```bash
bash run_all.sh
```

**NOTE!**

If you modify the dataset, remember to clear your cache!
Otherwise it will find the old dataset and not re-tokenize/re-group the text.

**IMPORTANT!**

Setting batch size over 1 ruins the performance on the `gsm` benchmark!


## TODO
- [ ] Generalise EWC model code
- [ ] Parallelize Fisher computation code
