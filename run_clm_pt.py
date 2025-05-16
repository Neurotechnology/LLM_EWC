from itertools import chain
from pathlib import Path
from typing import Optional, List, Dict, Any, Mapping
import logging
import math
import os
import re
import sys


from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        )
from peft.tuners.lora import LoraLayer
from sklearn.metrics import accuracy_score
from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        CONFIG_MAPPING,
        HfArgumentParser,
        LlamaForCausalLM,
        LlamaTokenizer,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
        is_torch_tpu_available,
        set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import (
        PREFIX_CHECKPOINT_DIR,
        get_last_checkpoint,
        )
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
import datasets
import numpy as np
import torch
import transformers

from gemma_ewc import GemmaEWC


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()
        # enable gradient checkpointing for memory efficiency
        # VYCKA:
        # Reentrant for passing multiple times over same param.
        #checkpointing_kwargs = {"use_reentrant": False}
        # Alternative
        # checkpointing_kwargs = {}
        #model.gradient_checkpointing_enable(
        #        # gradient_checkpointing_kwargs = checkpointing_kwargs,
        #        gradient_checkpointing_kwargs={'use_reentrant': False}
        #        )
        raise Exception('Uh oh...')

    return model


def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_token: Optional[str] = field(
        default=None, metadata={"help": "Dataset authorization token for hugging face)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    EWC_lambda: Optional[float] = field(default=1e12)
    EWC_param_dir: Optional[str] = field(default=None)


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    tokenizer.add_eos_token = False # Changed here
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def clean_wikipedia_text(text):
        pattern = r'\{\|.*?\|\}'
        if isinstance(text, str):  # If text is a single string
            cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
            return cleaned_text
        elif isinstance(text, list):  # If text is a list of strings
            cleaned_texts = [re.sub(pattern, '', t, flags=re.DOTALL) for t in text]
            return cleaned_texts
        else:
            raise ValueError("Input must be a string or a list of strings")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            #examples["text"] = [text.replace("\\n", "\n") for text in examples["text"]]
            cleaned_texts = clean_wikipedia_text(examples["text"])
            output = tokenizer(cleaned_texts)
            #output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Define the map function
    def filter_keys(batch):
        return {'input_ids': batch['input_ids'], 'labels': batch['labels'], 'attention_mask': batch['attention_mask']}
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result
    

    
    def group_texts_2(dataset, block_size=128, window_size=50):
        """
        Groups tokenized texts into chunks with a maximum block size, with overlap defined by window size.

        Args:
        - dataset: A Hugging Face Dataset object containing tokenized texts.
        - block_size: Maximum size of each block of tokens.
        - window_size: Number of tokens for overlap between blocks.

        Returns:
        - A new dataset with texts grouped according to the specified block size and window size.
        """
        # Initialize lists to hold the grouped tokens
        grouped_inputs = []

        # Iterate over each example in the dataset
        for example in dataset:
            tokens = example['input_ids']
            total_length = len(tokens)
            # Start index for each chunk
            start_index = 0
            while start_index < total_length:
                end_index = min(start_index + block_size, total_length)
                # Append the chunk to the grouped inputs
                grouped_inputs.append(tokens[start_index:end_index])
                # Code readibility is important
                if end_index < total_length:
                    start_index = end_index - window_size
                else:
                    start_index = total_length

        # Create a new dataset from the grouped inputs
        grouped_dataset = Dataset.from_dict({'input_ids': grouped_inputs, 'labels': grouped_inputs})
        return grouped_dataset
    
    lm_datasets_ls = []
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        
        if not os.path.isdir(data_args.dataset) or True:

            datsets_list = data_args.dataset.split(",")

            for dataset_name in datsets_list:
                filename = dataset_name
                if os.path.isdir(filename):
                    filename = os.path.basename(filename)
                cache_path = os.path.join(data_args.data_cache_dir, filename+f"_{block_size}")       
                os.makedirs(cache_path, exist_ok=True)                
                try:
                    logger.info(f'Loading dataset from {cache_path}')
                    processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)

                except Exception:
                    cache_dir = os.path.join(data_args.data_cache_dir, filename+f"_text_{block_size}")

                    if os.path.isdir(dataset_name) and False:
                        dataset = datasets.load_from_disk(dataset_name)
                        dataset = dataset['train']
                    else:
                    
                        dataset = load_dataset(dataset_name,
                                "lt",
                                token=data_args.dataset_token,                                 
                                streaming=False, # optional
                                               split="train",
                                cache_dir=data_args.data_cache_dir) # optional, but the dataset only has a train split
                        
                    existing_columns = dataset.column_names
                    columns_to_remove = ["text", 'id', 'url', 'title', 'meta']
                    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]

                    tokenized_dataset = dataset.map(
                                tokenize_function,
                                batched=True,
                                num_proc=data_args.preprocessing_num_workers,
                                remove_columns=columns_to_remove,
                                load_from_cache_file=True,
                                keep_in_memory=False,
                                desc="Running tokenizer on dataset",
                            )
                    
                    grouped_datasets = group_texts_2(
                            tokenized_dataset,
                            block_size=block_size,
                            window_size=block_size//2)
                
                    processed_dataset = grouped_datasets
                    processed_dataset.save_to_disk(cache_path)
                lm_datasets_ls.append(processed_dataset)
            
        else:
            path = Path(data_args.dataset)
            files = [file.name for file in path.glob("*.txt")]
            if training_args.debug_mode is True:
                files = [files[0]]
            file_count = len(files)
            
            for idx, file in enumerate(files):
                data_file = os.path.join(path, file)
                filename = ''.join(file.split(".")[:-1])
                cache_path = os.path.join(data_args.data_cache_dir, filename+f"_{block_size}")
                os.makedirs(cache_path, exist_ok=True)
                try:
                    processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                    #logger.info(f'training datasets-{filename} has been loaded from disk')
                    if idx % 100 == 0:
                        logger.info(f'loading datasets - {idx} of {file_count}')
                except Exception:
                    cache_dir = os.path.join(data_args.data_cache_dir, filename+f"_text_{block_size}")
                    os.makedirs(cache_dir, exist_ok=True)
                    raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                    logger.info(f"{file} has been loaded")
                    tokenized_dataset = raw_dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns="text",
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names = {k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                        desc="Running tokenizer on dataset",
                    )
                    grouped_datasets = tokenized_dataset.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names = {k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                        desc=f"Grouping texts in chunks of {block_size}",
                    )
                    processed_dataset = grouped_datasets
                    processed_dataset.save_to_disk(cache_path)
                lm_datasets_ls.append(processed_dataset['train'])
        lm_datasets = concatenate_datasets(lm_datasets_ls)
        if data_args.validation_split_percentage > 0:
            lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)

    if training_args.do_train:
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            bnb_8bit_compute_dtype=torch.bfloat16,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            #bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None 
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        logger.info(f"{torch_dtype} {load_in_4bit} {load_in_8bit}")
        logger.info(f"{quantization_config}")
        device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
        #model = AutoModelForCausalLM.from_pretrained(
        #    model_args.model_name_or_path,
        #    from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #    config=config,
        #    cache_dir=model_args.cache_dir,
        #    revision=model_args.model_revision,
        #    use_auth_token=True if model_args.use_auth_token else None,
        #    torch_dtype=torch_dtype,
        #    #low_cpu_mem_usage=True,
        #    #device_map=device_map,
        #    load_in_4bit=load_in_4bit,
        #    load_in_8bit=load_in_8bit,
        #    quantization_config=quantization_config,
        #)
        model = GemmaEWC.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            EWC_lambda = training_args.EWC_lambda,
            EWC_param_dir = training_args.EWC_param_dir,
        )
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")

    if model_vocab_size != tokenizer_vocab_size:
        logger.info(f"Resize model vocab size to {tokenizer_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    if training_args.do_train:

        trainer.train()
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
