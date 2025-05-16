docker run \
	--shm-size=20gb \
	--gpus all \
	-v $(pwd):/LLM_EWC \
	-v <VOLUME_LOCAL>:<VOLUME_DOCKER> \
	-p 5001:5000 \
	-it llm_ewc
