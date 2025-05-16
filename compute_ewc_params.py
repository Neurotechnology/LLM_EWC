
from collections import defaultdict
import argparse
import pickle

import datasets
import torch
from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        )
from tqdm import tqdm



MMLU_question_keys = [
        'A. ',
        'B. ',
        'C. ',
        'D. ',
        ]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def generate_prompt_mmlu(item: dict) -> str:
    output = []
    for i in range(len(item['subject'])):
        subject = item['subject'][i]
        question = item['question'][i]
        prompt = 'The following is a multiple choice question (with answers) about {}. Respond only with the letter corresponding to the right answer.'.format(format_subject(subject))
        prompt += 'Example:\n'
        prompt += 'How much is 2+2?\n'
        prompt += 'A. 0\n'
        prompt += 'B. 2\n'
        prompt += 'C. 4\n'
        prompt += 'D. 9\n'
        prompt += 'Answer:\n'
        prompt += 'C\n'
        prompt += question
        prompt += '\n'
        for j in range(4):
            prompt += MMLU_question_keys[j]
            prompt += item['choices'][j][i]
            prompt += '\n'

        prompt += 'Answer:\n'
        output.append(prompt)
    return output


def main(args):
    PRETRAINED_MODEL = args.pretrained_model_dir
    CACHE_DIR = args.cache
    BATCH_SIZE = args.batch_size

    loss_fun = torch.nn.CrossEntropyLoss()

    # Load model
    config_kwargs = {
        "cache_dir": CACHE_DIR,
        "revision": "main",
        "use_auth_token": None,
    }
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        from_tf=bool(".ckpt" in PRETRAINED_MODEL),
        config=config,
        cache_dir=CACHE_DIR,
        revision="main",
        use_auth_token= None,
        torch_dtype=torch.float16,
        #low_cpu_mem_usage=True,
        #device_map=device_map,
        load_in_4bit=False,
        load_in_8bit=False,
        quantization_config=None,
        ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    dataset_labels = [
            ("cais/mmlu", "all", "test",),
            #("openai/gsm8k", "main", "test",), # NOTE would love to implement these
            #("truthfulqa/truthful_qa", "multiple_choice", "validation",),
            #("winogrande", "winogrande_xl", "test",),
            #("allenai/ai2_arc", "ARC-Easy", "test",),
            #("Rowan/hellaswag", "test",),
            #("facebook/belebele", "eng_Latn", "test",),
            ]
    dsets = []
    for label in dataset_labels:
        print(label)
        config = label[:-1]
        split = label[-1]
        dset = datasets.load_dataset(
                *config,
                split=split,
                cache_dir=CACHE_DIR)
        fisher_matrix = defaultdict(float)
        dload = torch.utils.data.DataLoader(
                dset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                )
        for index, item in enumerate(tqdm(dload)):
            questions = generate_prompt_mmlu(item)
            tokens = tokenizer(
                    questions,
                    return_tensors='pt',
                    max_length=1024,
                    truncation=True,
                    padding = 'longest',
                    ).to('cuda')
            # For logits
            # pass through model
            output = model(
                    **tokens,
                    num_logits_to_keep=1,
                    )
            answers = []
            for i in range(len(item['answer'])):
                answers.append(MMLU_question_keys[item['answer'][i]][0])
            answer_token = tokenizer(
                    answers,
                    return_tensors="pt").to('cuda')
            answer_ID = answer_token['input_ids'][:,1]
            loss = loss_fun(output.logits[:,0,:], answer_ID)
            # compute gradient of loglikelihoods wrt weights, save
            loss.backward()
            for n,p in model.named_parameters():
                fisher_matrix[n] = fisher_matrix[n] + p.grad.data.detach().cpu() **2
            model.zero_grad()
        norm_fisher_matrix = {n: fisher_matrix[n] / len(dload) for n in fisher_matrix}
        # Clean fisher matrix from infs
        for param_name, param_val in norm_fisher_matrix.items():
            inf_indices = param_val == torch.inf
            norm_fisher_matrix[param_name][inf_indices] = args.inf_cap

        model = model.to('cpu') # Otherwise it'll load them to first gpu on loading
        param_tuples = [(param_name, param) for param_name, param in model.named_parameters()]
        with open(args.output, 'wb') as f:
            pickle.dump((param_tuples, norm_fisher_matrix),f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Utility to compute Fisher matrices from MMLU',
            )
    parser.add_argument(
            'pretrained_model_dir',
            type=str,
            help='Location of pretrained model to compute Fisher matrix for'
            )
    parser.add_argument(
            '--output',
            type=str,
            default='EWC_params.pkl',
            )
    parser.add_argument(
            '--cache',
            help='Location of cache',
            type=str,
            default='.cache/',
            )
    parser.add_argument(
            '--batch-size',
            type=int,
            default=4,
            )
    parser.add_argument(
            '--inf-cap',
            type=float,
            default=1e2,
            help='The value to replace `inf`s in the Fisher matrix.',
            )
    args = parser.parse_args()
    main(args)

