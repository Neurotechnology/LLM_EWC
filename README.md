# Neuro EWC on LLM

This is the implementation code for the preprint [Elastic Weight Consolidation for Full-Parameter Continual Pre-Training of Gemma2](https://arxiv.org/abs/2505.05946).

## Installation

Modify `run_docker.sh` with any additional volumes where data is stored
by inserting the following lines:

`-v <VOLUME_LOCAL>:<VOLUME_DOCKER> \`

Then build the image and run container:
```bash
docker build -t llm_ewc .
./run_docker.sh
```

## Running

### Computing fisher matrix

We created a handy script that you need to modify at `compute_ewc_params.sh`.
Then the params can be computed using the following:

```bash
compute_ewc_params.sh
```

## TODO

I will move over code from the local repo to this one because it has a lot of
naughty stuff leftover, such as access tokens, swearwords and maybe even some
info we may not want to leak.

