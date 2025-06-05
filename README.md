# Neuro EWC on LLM

This is the implementation code for the preprint [Full-Parameter Continual Pretraining of Gemma2: Insights into Fluency and Domain Knowledge](https://arxiv.org/abs/2505.05946).

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

To run training, you first need to export relevant environment variables.
`DATASET_TOKEN` is the dataset token from huggingface, while `TRANSFORMERS_CACHE` is the cache location used by the transformers library.
```bash
export DATASET_TOKEN=hf_tokentokentokentoken
export TRANSFORMERS_CACHE=<...>/.cache
```
The exporting can be done automatically while running the container.
The `run_docker.sh` file needs the following argument added:
```bash
+       --env-file .env \
```
And the `.env`. file should look like the following:
```
DATASET_TOKEN=hf_tokentokentokentoken
TRANSFORMERS_CACHE=<...>/.cache
```

Then, after editing paths in the file `run_gemma2-2b_culturaX.sh`, the training can be run:

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

## Evaluation

We evaluated the models using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).
You will need to install it yourself by following the instructions.
An example shell script is provided at `eval.sh`

**IMPORTANT!**

Setting batch size over 1 ruins the performance on the `gsm` benchmark!


## TODO
- [ ] Generalise EWC model code
- [ ] Parallelize Fisher computation code
