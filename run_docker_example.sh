docker run \
	--shm-size=20gb \
	--gpus all \
	-v $(pwd):/LLM_EWC \
	-v <VOLUME_LOCAL>:<VOLUME_DOCKER> \
	-p 5001:5000 \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/group:/etc/group:ro \
	--env-file .env \
	--user $(id -u):$(id -g) \
	-it llm_ewc
