import os
import sys
import datasets
import json
import torch
import pandas as pd
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from infbench_utils import TASK_TO_PATH, TASK_TO_MAX_NEW_TOKENS, get_score_one, create_prompt, get_answer

def process_infbench(data, indices, tokenizer, task, max_length=100000):
    outputs = {'input_ids': [], 'attention_mask': [], "index": [], "answer": []}

    data = pd.DataFrame(dict(data)).to_dict(orient="records")

    for sample, index in zip(data, indices):
        prompt = create_prompt(sample, task, 'mistral')
        answer = get_answer(sample, task)

        
        tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        encoded = tokenizer(prompt)

        outputs["input_ids"].append(encoded["input_ids"])
        outputs["attention_mask"].append(encoded["attention_mask"])
        outputs["index"].append(index)
        outputs["answer"].append(answer)

    return outputs

@torch.no_grad()
def main(args):
    result_dir = "../workspace/infbench_pred/"
    result_dir += (args.model.split("/")[-1]+'_'+args.edition)
    if args.edition == "promptdistill" and (not args.use_pooling):
        result_dir += "_nopool"
    if args.edition == "promptdistill" and args.cache_truncation_end != 0:
        result_dir = f"{result_dir}_cache{args.cache_truncation_end}"
    if args.edition == "promptdistill":
        for i in range(len(args.selection_layers)):
            result_dir = f"{result_dir}_layer{args.selection_layers[i]}_top{args.n_topks[i]}"
    elif args.edition == "gemfilter":
        result_dir = f"{result_dir}_layer{args.selection_layers[0]}_top{args.n_topks[0]}"
    elif args.edition != "default":
        result_dir = f"{result_dir}_top{args.n_topks[0]}"
    os.makedirs(result_dir, exist_ok=True)
    
    if args.edition == "promptdistill":
        model = PromptDistill.from_pretrained(args.model,attn_implementation='flash_attention_2',torch_dtype=torch.bfloat16).to('cuda').eval()
        model.model.use_pooling = args.use_pooling
        model.model.n_topks = args.n_topks
        model.model.selection_layers = args.selection_layers
    elif args.edition == "default":
        model = AutoModelForCausalLM.from_pretrained(args.model,attn_implementation='flash_attention_2',torch_dtype=torch.bfloat16).to('cuda').eval()
    else:
        model = load_model(args.model, modified=args.edition, torch_dtype=torch.float16, flash_attention_2=(args.edition != 'h2o'))
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.requires_grad_(False)

    
    max_length = 100000
    if args.model_type == "llama":
        max_length = 100000
    if args.model_type == "phi3":
        max_length = 50000
    if args.model_type == "qwen2":
        max_length = 31500
    if args.edition == 'h2o':
        max_length = 8192

    tasks = ["passkey","number_string","kv_retrieval","longbook_sum_eng","longbook_choice_eng","longbook_qa_eng","longbook_qa_chn","longdialogue_qa_eng","math_find","code_run","code_debug"]

    all_datasets = {}

    for task in tasks:
        process_fn = partial(
            process_infbench, 
            tokenizer=tokenizer,
            task=task,
            max_length=max_length
        )
        path = os.path.join("../workspace/rawdata/infbench", TASK_TO_PATH[task])
        raw_dataset = datasets.load_dataset("json", data_files=path, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, batch_size=10, with_indices=True, remove_columns=raw_dataset.column_names)

        all_datasets[task] = dataset


    metrics = {"name":result_dir.split('../workspace/infbench_pred/')[1]}

    for i, (task, dataset) in enumerate(all_datasets.items()):
        result_path = os.path.join(result_dir, f"{task}.json")
        # get answers in advance
        labels = dataset["answer"]
        dataset = dataset.remove_columns(["answer"])

        indices = []
        preds = []
        max_new_tokens = TASK_TO_MAX_NEW_TOKENS[task]

        for inputs in dataset:
            index = inputs.pop("index")
            inputs["input_ids"] = torch.tensor([inputs["input_ids"]],dtype=torch.int64, device=model.device)
            inputs['attention_mask'] = torch.tensor([inputs['attention_mask']],dtype=torch.int64, device=model.device)
            input_length = inputs["input_ids"].shape[1]

            if args.edition in ['gemfilter','snapkv','h2o']:
                set_topk(model, args.n_topks[0], mode=args.edition)
                if args.edition == 'gemfilter':
                    response = my_greedy_generate_selection(
                    inputs['input_ids'], inputs['attention_mask'], model, tokenizer, max_gen_len=max_new_tokens, select_layer_idx=args.selection_layers[0])
                else:
                    response = my_greedy_generate_standard(inputs['input_ids'], inputs['attention_mask'], model, tokenizer, max_gen_len=max_new_tokens)
            else:
                prompt_length = inputs['input_ids'].shape[1]
                if args.edition == "promptdistill":
                    model.model.cache_truncation_end = args.cache_truncation_end
                with torch.no_grad():
                    outputs = model(inputs['input_ids'], attention_mask=inputs["attention_mask"], output_last_logits_only=True)
                if args.edition == "promptdistill":
                    model.model.cache_truncation_end = 0
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids = [pred_token_idx.item()]
                for _ in range(max_new_tokens):
                    position_ids = torch.tensor([[prompt_length+_]],dtype=torch.int64, device=pred_token_idx.device)
                    with torch.no_grad():
                        outputs = model(input_ids=pred_token_idx, position_ids = position_ids, past_key_values=past_key_values)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    if pred_token_idx == tokenizer.eos_token_id:
                        break
                    generated_ids.append(pred_token_idx.item())
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            #output = model.generate(**x,max_new_tokens=max_new_tokens)

            preds.append(response)
            indices.append(index)

        scores = []
        for label, pred in tqdm(zip(labels, preds)):
            score = get_score_one(pred, label, task, None)
            scores.append(score)
        score = round(sum(scores) / len(scores), 4)

        metrics[task] = score
            
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(score, ensure_ascii=False) + "\n")
            for index, pred, label in zip(indices, preds, labels):
                item = {
                    "index": index,
                    "pred": pred,
                    "label": label,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct', type=str,help="The model used for evaluation")
    parser.add_argument('--model-type', default='llama', choices=['llama','phi3','qwen2'], type=str, help="Specify which model file in models directory should be used, also specify max input length")
    parser.add_argument('--edition', default="promptdistill", choices=["default", "promptdistill",'gemfilter','snapkv','h2o'],type=str, help="The method used to edit the original model, default is original model without edition")
    parser.add_argument('--tokenizer', default='', type=str, help="The tokenizer used for evaluation, if same as model, can be skipped")
    parser.add_argument('--use-pooling', default=True, action='store_true', help="Whether to use pooling after selection weights calculated")
    parser.add_argument('--n-topks', default=[2048], type=int, nargs='+', help="List of numbers of selected tokens correspond to selection layers")
    parser.add_argument('--selection-layers', default=[13], type=int, nargs='+', help="List of selection layers")
    parser.add_argument('--cache-truncation-end', default=0, type=int, help="Variable tt in the paper, decide how many times to use cache truncation (after each selection layer), set 0 if don't use cache truncation, set -1 if use cache truncation for all selection times")
    parser.add_argument('--no-use-pooling', dest='use_pooling', action='store_false')
    args = parser.parse_args()
    if args.tokenizer == "":
        args.tokenizer = args.model

    os.makedirs("../workspace", exist_ok=True)
    os.makedirs("../workspace/infbench_pred", exist_ok=True)
    if args.edition in ['gemfilter','snapkv','h2o']:
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..','baselines')))
        from my_utils.load_model import load_model
        from my_utils.my_generation import set_topk, my_greedy_generate_selection, my_greedy_generate_standard

    if args.edition == "promptdistill":
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
        if args.model_type == "llama":
            from models.llama import PromptDistill
        if args.model_type == "phi3":
            from models.phi3 import PromptDistill
        if args.model_type == "qwen2":
            from models.qwen2 import PromptDistill
        
        
    main(args)
