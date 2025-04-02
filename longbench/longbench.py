import os
import sys
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def get_pred(args,dataset,data, max_length, max_gen, prompt_format, model, tokenizer,  out_path):
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)

        if args.edition in ['gemfilter','snapkv','h2o']:
            set_topk(model, args.n_topks[0], mode=args.edition)
            if args.edition == 'gemfilter':
                response = my_greedy_generate_selection(
                inputs['input_ids'], inputs['attention_mask'], model, tokenizer, max_gen_len=max_gen, select_layer_idx=args.selection_layers[0])
            else:
                response = my_greedy_generate_standard(inputs['input_ids'], inputs['attention_mask'], model, tokenizer, max_gen_len=max_gen)
        else:
            prompt_length = inputs['input_ids'].shape[1]
            if args.edition == "promptdistill":
                model.model.cache_truncation_end = args.cache_truncation_end
            with torch.no_grad():
                outputs = model(inputs['input_ids'], attention_mask=inputs["attention_mask"])
            if args.edition == "promptdistill":
                model.model.cache_truncation_end = 0
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = [pred_token_idx.item()]
            for _ in range(max_gen):
                position_ids = torch.tensor([[prompt_length+_]],dtype=torch.int64, device=pred_token_idx.device)
                with torch.no_grad():
                    outputs = model(input_ids=pred_token_idx, position_ids = position_ids, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if pred_token_idx == tokenizer.eos_token_id:
                    break
                generated_ids.append(pred_token_idx.item())
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": response, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_name(args):
    if args.e:
        path = "../workspace/longbench_pred_e/"
    else:
        path = "../workspace/longbench_pred/"
    name = (args.model.split("/")[-1]+'_'+args.edition)
    if args.edition == "promptdistill" and (not args.use_pooling):
        name += "_nopool"
    if args.edition == "promptdistill" and args.cache_truncation_end != 0:
        name = f"{name}_cache{args.cache_truncation_end}"
    if args.edition == "promptdistill":
        for i in range(len(args.selection_layers)):
            name = f"{name}_layer{args.selection_layers[i]}_top{args.n_topks[i]}"
    elif args.edition == "gemfilter":
        name = f"{name}_layer{args.selection_layers[0]}_top{args.n_topks[0]}"
    elif args.edition != "default":
        name = f"{name}_top{args.n_topks[0]}"
    path = path+name+"/"
    return name, path

def get_data_folder_name(args,dataset):
    if args.e:
        data = load_dataset('THUDM/LongBench',f"{dataset}_e", split='test',trust_remote_code=True)
        path = "../workspace/longbench_pred_e/"
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test',trust_remote_code=True)
        path = "../workspace/longbench_pred/"
    path += (args.model.split("/")[-1]+'_'+args.edition)
    if args.edition == "promptdistill" and (not args.use_pooling):
        path += "_nopool"
    if args.edition == "promptdistill" and args.cache_truncation_end != 0:
        path = f"{path}_cache{args.cache_truncation_end}"
    if args.edition == "promptdistill":
        for i in range(len(args.selection_layers)):
            path = f"{path}_layer{args.selection_layers[i]}_top{args.n_topks[i]}"
    elif args.edition == "gemfilter":
        path = f"{path}_layer{args.selection_layers[0]}_top{args.n_topks[0]}"
    elif args.edition != "default":
        path = f"{path}_top{args.n_topks[0]}"
        
    os.makedirs(path, exist_ok=True)
    path = f"{path}/{dataset}.jsonl"
        
    return data, path
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct', type=str, help="The model used for evaluation")
    parser.add_argument('--model-type', default='llama', choices=['llama','phi3','qwen2'], type=str, help="Specify which model file in models directory should be used, also specify max input length")
    parser.add_argument('--edition', default="promptdistill", choices=["default", "promptdistill",'gemfilter','snapkv','h2o'],type=str, help="The method used to edit the original model, default is original model without edition")
    parser.add_argument('--tokenizer', default='', type=str, help="The tokenizer used for evaluation, if same as model, can be skipped")
    parser.add_argument('--use-pooling', default=True, action='store_true', help="Whether to use pooling after selection weights calculated")
    parser.add_argument('--n-topks', default=[1024], type=int, nargs='+', help="List of numbers of selected tokens correspond to selection layers")
    parser.add_argument('--selection-layers', default=[13], type=int, nargs='+', help="List of selection layers")
    parser.add_argument('--cache-truncation-end', default=0, type=int, help="Variable tt in the paper, decide how many times to use cache truncation (after each selection layer), set 0 if don't use cache truncation, set -1 if use cache truncation for all selection times")
    parser.add_argument('--no-use-pooling', dest='use_pooling', action='store_false')
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    args = parser.parse_args()
    if args.tokenizer == "":
        args.tokenizer = args.model

    os.makedirs("../workspace", exist_ok=True)
    os.makedirs("../workspace/longbench_pred", exist_ok=True)
    os.makedirs("../workspace/longbench_pred_e", exist_ok=True)
    seed_everything(42)
    
    if args.edition in ['gemfilter','snapkv','h2o']:
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'baselines')))
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

    max_length = 128000
    if args.model_type == "qwen2":
        max_length = 32200
    if args.edition == 'h2o':
        max_length = 8192
        
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "passage_count", "passage_retrieval_en"]
    dataset2prompt = json.load(open("dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("dataset2maxlen.json", "r"))

    
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
    for dataset in datasets:
        print(dataset+' started')
        data, out_path = get_data_folder_name(args,dataset)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data = [data_sample for data_sample in data]
        get_pred(args,dataset,data, max_length,max_gen, prompt_format, model, tokenizer, out_path)
        
    scores = dict()
    name, path = get_name(args)
    scores["name"] = name
    
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    with open(f"{path}result.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)