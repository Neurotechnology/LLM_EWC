# Neuro EWC on LLM

This is the implementation code for the preprint [Elastic Weight Consolidation for Full-Parameter Continual Pre-Training of Gemma2](https://arxiv.org/abs/2505.05946).

## Installation

Modify `run_docker.sh` with any additional volumes where data is stored
by inserting the following lines:

`-v <VOLUME_LOCAL>:<VOLUME_DOCKER> \`

Then build the image and run container:
```bash
docker build -t llm_ewc .
bash run_docker.sh
```

## Running

### Computing fisher matrix

We created a handy script that you need to modify at `compute_ewc_params.sh`.
Then the params can be computed using the following:

```bash
bash compute_ewc_params.sh
```

To run training, you first need to export your huggingface dataset token:
```bash
export DATASET_TOKEN=hf_tokentokentokentoken
```

Then, after editing paths in the file `run_gemma2-2b_culturaX.sh`, the training can be run:

```bash
bash run_gemma2-2b_culturaX.sh 1e2
```

Or several of them can be run:

```bash
bash run_all.sh
```

NOTE!
If you modify the dataset, remember to clear your cache!
Otherwise it will find the old dataset and not re-tokenize/re-group the text.

## TODO
- [ ] Generalise EWC model code
- [ ] Parallelize Fisher computation code
