lr=2e-4
EWC_lambda=1e2
EWC_param_dir=<...>/EWC_params_test.pkl
pretrained_model=<...>/models/gemma-2-2b-it/
tokenizer_path=<...>/models/gemma-2-2b-it/
data_cache=<...>/.cache
output_dir=<...>/gemma2-2b-it/${EWC_lambda}
dataset=uonlp/CulturaX
per_device_train_batch_size=4
gradient_accumulation_steps=4
block_size=512
nproc_per_node=8 # How many GPUs this is run on
deepspeed_config_file=ds_no_offload.json

torchrun --nnodes 1 --nproc_per_node ${nproc_per_node} run_clm_pt.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset ${dataset} \
    --dataset_token ${DATASET_TOKEN} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 24 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False\
    --EWC_lambda ${EWC_lambda}\
    --EWC_param_dir ${EWC_param_dir}
