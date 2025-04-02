import os 
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

import time
import torch
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def visualize(args,model_name,figname):
    plt.rcParams.update({'font.size': 23})
    # Path to the directory containing JSON results
    folder_path = "../workspace/needle_pred/" + model_name + "/"
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(expected_answer)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    locations = list(df["Context Length"].unique())
    locations.sort()

    #print(df.head())
    #df = df.drop(df[df['Context Length'] > length_limit].index)
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    plt.title(f'Pressure Testing {figname} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    os.makedirs("../workspace/needle_eval", exist_ok=True)
    save_path = "../workspace/needle_eval/%s.pdf" % model_name
    print("saving at %s" % save_path)
    plt.tight_layout(pad=1.01)
    plt.savefig(save_path)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 40,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percent_interval_type = "linear",
                 args=None,
                 save_results = True,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True
                 ):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        self.args = args

        self.name = args.model.split("/")[-1]+'_'+args.edition
        if args.edition == "promptdistill" and (not args.use_pooling):
            self.name += "_nopool"
        if args.edition == "promptdistill" and args.cache_truncation_end != 0:
            self.name = f"{self.name}_cache{args.cache_truncation_end}"
        if args.edition == "promptdistill":
            for i in range(len(args.selection_layers)):
                self.name = f"{self.name}_layer{args.selection_layers[i]}_top{args.n_topks[i]}"
        elif args.edition == "gemfilter":
            self.name = f"{self.name}_layer{args.selection_layers[0]}_top{args.n_topks[0]}"
        elif args.edition != "default":
            self.name = f"{self.name}_top{args.n_topks[0]}"

        self.figname = args.model.split("/")[-1]+'_'
        if args.edition == "default":
            self.figname += "AllKV"
        if args.edition == "promptdistill":
            self.figname += "PromptDistill"
        if args.edition == "gemfilter":
            self.figname += "GemFilter"
        if args.edition == "snapkv":
            self.figname += "SnapKV"
        if args.edition == "h2o":
            self.figname += "H2O"
        if args.edition == "promptdistill" and (not args.use_pooling):
            self.figname += "_nopool"
        if args.edition == "promptdistill":
            for i in range(len(args.selection_layers)):
                self.figname = f"{self.figname}_layer{args.selection_layers[i]}_top{args.n_topks[i]}"
        elif args.edition == "gemfilter":
            self.figname = f"{self.figname}_layer{args.selection_layers[0]}_top{args.n_topks[0]}"
        elif args.edition != "default":
            self.figname = f"{self.figname}_top{args.n_topks[0]}"

        if self.save_results:
            os.makedirs('../workspace/needle_pred/' + self.name, exist_ok=True)

        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)

        if document_depth_percent_interval_type == 'linear':
            self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
        elif document_depth_percent_interval_type == 'sigmoid':
            self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]

        print("loading from %s" % args.model)

        if args.edition == "promptdistill":
            model = PromptDistill.from_pretrained(args.model,attn_implementation='flash_attention_2',torch_dtype=torch.bfloat16).to('cuda').eval()
            model.model.use_pooling = args.use_pooling
            model.model.n_topks = args.n_topks
            model.model.selection_layers = args.selection_layers
        elif args.edition == "default":
            model = AutoModelForCausalLM.from_pretrained(args.model,attn_implementation='flash_attention_2',torch_dtype=torch.bfloat16).to('cuda').eval()
        else:
            model = load_model(args.model, modified=args.edition, torch_dtype=torch.float16, flash_attention_2=(args.edition != 'h2o'))
            
        self.model = model
        self.enc = AutoTokenizer.from_pretrained(args.tokenizer)


    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    

    def run_test(self):
        for context_length in self.context_lengths:
            if context_length < self.args.s_len or context_length > self.args.e_len: continue
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent)


    def evaluate_and_log(self, context_length, depth_percent):
        
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")
        

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        test_start_time = time.time()
        
        prompt = self.enc(prompt, return_tensors="pt")
        input_ids = prompt['input_ids'].to(self.model.device)
        attn_mask = prompt["attention_mask"].to(self.model.device)

        if self.args.edition in ['gemfilter','snapkv','h2o']:
            set_topk(self.model, self.args.n_topks[0], mode=self.args.edition)
            if self.args.edition == 'gemfilter':
                response = my_greedy_generate_selection(
                input_ids, attn_mask, self.model, self.enc, max_gen_len=50, select_layer_idx=self.args.selection_layers[0])
            else:
                response = my_greedy_generate_standard(input_ids, attn_mask, self.model, self.enc, max_gen_len=50)
        else:
            prompt_length = input_ids.shape[1]
            if self.args.edition == "promptdistill":
                self.model.model.cache_truncation_end = self.args.cache_truncation_end
            with torch.no_grad():
                outputs = self.model(input_ids, attn_mask)
            if self.args.edition == "promptdistill":
                self.model.model.cache_truncation_end = 0
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = [pred_token_idx.item()]
            for _ in range(50):
                position_ids = torch.tensor([[prompt_length+_]],dtype=torch.int64, device=pred_token_idx.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=pred_token_idx, position_ids = position_ids, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if pred_token_idx == self.enc.eos_token_id:
                    break
                generated_ids.append(pred_token_idx.item())
            response = self.enc.decode(generated_ids, skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score = scorer.score(self.needle, response)['rouge1'].fmeasure*10

        results = {
            'model' : self.args.model,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.name}_len_{context_length}_depth_{int(depth_percent*100)}'
            
        if self.save_results:
            results_dir = '../workspace/needle_pred/' + self.name

            # Save the result to file for retesting
            p = f'{results_dir}/{context_file_location}_results.json'
            #print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        results_dir = '../workspace/needle_pred/' + self.name
        #print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.args.model
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.enc.encode(self.needle)
        tokens_context = self.enc.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is 
            period_tokens = self.enc.encode('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.enc.decode(tokens_new_context)
        return new_context
        

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while len(self.enc.encode(context)) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context



    def encode_and_trim(self, context, context_length):
        tokens = self.enc.encode(context)
        if len(tokens) > context_length:
            context = self.enc.decode(tokens[:context_length])
        return context
    
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.args.model}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', type=int, default=0, help="No context length shorter than this will be evaluated")
    parser.add_argument('-e', '--e_len', type=int, default=128000, help="No context length longer than this will be evaluated")
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct', type=str,help="The model used for evaluation")
    parser.add_argument('--model-type', default='llama', choices=['llama','phi3','qwen2'], type=str, help="Specify which model file in models directory should be used, also specify max input length")
    parser.add_argument('--edition', default="promptdistill", choices=["default", "promptdistill",'gemfilter','snapkv','h2o'],type=str, help="The method used to edit the original model, default is original model without edition")
    parser.add_argument('--tokenizer', default='', type=str, help="The tokenizer used for evaluation, if same as model, can be skipped")
    parser.add_argument('--use-pooling', default=True, action='store_true', help="Whether to use pooling after selection weights calculated")
    parser.add_argument('--n-topks', default=[1024], type=int, nargs='+', help="List of numbers of selected tokens correspond to selection layers")
    parser.add_argument('--selection-layers', default=[13], type=int, nargs='+', help="List of selection layers")
    parser.add_argument('--cache-truncation-end', default=0, type=int, help="Variable tt in the paper, decide how many times to use cache truncation (after each selection layer), set 0 if don't use cache truncation, set -1 if use cache truncation for all selection times")
    parser.add_argument('--no-use-pooling', dest='use_pooling', action='store_false')
    args = parser.parse_args()
    if args.tokenizer == "":
        args.tokenizer = args.model
    
    os.makedirs("../workspace", exist_ok=True)
    os.makedirs("../workspace/needle_pred", exist_ok=True)
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
    ht = LLMNeedleHaystackTester(args=args)

    if ht.print_ongoing_status:
        ht.print_start_test_summary()
    ht.run_test()
    
    visualize(args,ht.name,ht.figname)