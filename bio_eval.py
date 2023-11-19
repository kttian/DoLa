# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

import time 
import ssl
import urllib.request
import zipfile

from dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 7
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"

def load_csv(file_path, is_gzip=False):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only
    
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        list_data = list(df['Question'])

    return list_data

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def get_names(split=None, n=400):
    names_file = '/iris/u/kattian/project_hallucination/sft_rl_0719/long_form_results/labels400/names400.txt'
    with open(names_file, 'r') as f:
        names = f.read().split('\n')
    if split is None:
        return names 
    elif split == "train":
        return names[:320]
    elif split == "test":
        return names[-64:]
    return names 

def get_few_shot_prompt():
    # instructions = ["LeBron James", "Hillary Clinton"]
    # biographies = ["LeBron James is a professional basketball player and one of the best in the NBA. He was born in Akron, Ohio on December 30, 1984. LeBron James' NBA career started with the Cleveland Cavaliers in 2003, and then he went to the Miami Heat in 2010. In 2014, James returned to Cleveland as a free agent. James is a 3x NBA Champion, 3x NBA Finals MVP, 2x NBA MVP, and 14x NBA All Star. He is the first athlete to be in the NBA Finals in three consecutive seasons. James now plays for the Cleveland Cavaliers, and has been nicknamed King James.",
    #                "Born in 1947, Hillary Clinton was the daughter of a distinguished politician, Hugh Rodham. Raised in an upper middle-class background, Hillary attended the Wellesley College in Massachusetts, followed by Yale Law School. In 1975, she married Bill Clinton, who later became the governor of Arkansas and the president of the United States. They have a daughter, Chelsea, born in 1980. Hillary Clinton served as the U.S. secretary of state, from 2009-2013, and is currently the senator of New York. She has authored the bestselling memoir 'Living History.'"]
    # few_shot_prompt = ""
    # for i in range(len(instructions)):
    #     few_shot_prompt += "NAME: " + instructions[i] + ". BIOGRAPHY: " + biographies[i] + "\n"
    few_shot_prompt = "NAME: Lebron James. BIOGRAPHY: LeBron James is a professional basketball player and one of the best in the NBA. He was born in Akron, Ohio on December 30, 1984. LeBron James' NBA career started with the Cleveland Cavaliers in 2003, and then he went to the Miami Heat in 2010. In 2014, James returned to Cleveland as a free agent. James is a 3x NBA Champion, 3x NBA Finals MVP, 2x NBA MVP, and 14x NBA All Star. He is the first athlete to be in the NBA Finals in three consecutive seasons. James now plays for the Cleveland Cavaliers, and has been nicknamed King James. "
    few_shot_prompt += "NAME: Hillary Clinton. BIOGRAPHY: Born in 1947, Hillary Clinton was the daughter of a distinguished politician, Hugh Rodham. Raised in an upper middle-class background, Hillary attended the Wellesley College in Massachusetts, followed by Yale Law School. In 1975, she married Bill Clinton, who later became the governor of Arkansas and the president of the United States. They have a daughter, Chelsea, born in 1980. Hillary Clinton served as the U.S. secretary of state, from 2009-2013, and is currently the senator of New York. She has authored the bestselling memoir 'Living History.' "
    return few_shot_prompt

def get_sft_prompt(name):
    return f"Here is a biography of {name}."

def build_prompt(input_name):
    demo = get_few_shot_prompt()
    input_text_prompt = demo + "NAME: " + input_name + ". BIOGRAPHY:"
    return input_text_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    start_time = time.time()

    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''
    # fp = os.path.join(args.data_path, 'TruthfulQA.csv')
    # if not os.path.exists(fp):
    #     download_url(
    #         'https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv', args.data_path)

    # list_data_dict = load_csv(fp)
    def get_train_names(n=288):
        names_file = '/iris/u/kattian/project_hallucination/direct-preference-optimization/kat_eval/create_test_split/train_names_286.txt'
        with open(names_file, 'r') as f:
            names = f.read().split('\n')
        return names[:n]

    def get_test_names(n=40):
        names_file = '/iris/u/kattian/project_hallucination/direct-preference-optimization/kat_eval/create_test_split/test_names_40.txt'
        with open(names_file, 'r') as f:
            names = f.read().split('\n')
        return names[:n]
    
    if args.split == "train":
        names = get_train_names(n=args.size)
    elif args.split == "test":
        names = get_test_names(n=args.size)
    # names = get_names(split=args.split)[:args.size]
    list_data_dict = names
    
    if args.debug:
        list_data_dict = list_data_dict[:10]

    print("list data dict:", list_data_dict)
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory, args.checkpoint_path)
    stop_word_list = ["NAME:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    
    result_dict = {'question': [], 'model_completion': [], "prompt": []}
    counter = 0
    for sample in tqdm(list_data_dict):
        if args.checkpoint_path is not None:
            input_text = get_sft_prompt(sample)
        else:
            input_text = build_prompt(sample)
        result_dict['prompt'].append(input_text)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty, 
                               mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers,
                               seed=args.seed)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        model_completion = model_completion.split(stop_word_list[0])[0]
        model_completion = model_completion.strip()
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        model_answer = model_completion
        result_dict['model_completion'].append(model_completion)
        result_dict['question'].append(sample)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        
        if counter < 10:
            print(f'Question: {sample}\n'
                f'Model Completion: {model_completion}\n')
        counter += 0

    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    result_dict['args'] = vars(args)
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    print(f"FINISHED DOLA IN TOTAL TIME: {time.time() - start_time:.3f} seconds")

    if args.do_rating:
        from tfqa_gpt3_rating import run_end2end_GPT3, load_json
        import json
        import warnings
        import openai
        import sys
        
        gpt3_config_file = args.gpt3_config
        if gpt3_config_file is None:
            warnings.warn("No GPT3 config set, skipping!", stacklevel=2)
            sys.exit(0)
        config = json.load(open(gpt3_config_file))
        openai.api_key = config['api_key']
        judge_name = config["gpt_truth"]
        info_name = config["gpt_info"]

        data = load_json(output_file)
        if args.debug:
            data['question'] = data['question'][:10]
            data['model_completion'] = data['model_completion'][:10]

        judge_scores, judge_accs = run_end2end_GPT3(data['question'], data['model_completion'], judge_name, info=False)
        info_scores, info_accs = run_end2end_GPT3(data['question'], data['model_completion'], info_name, info=True)

        avg_judge_score = sum(judge_scores) / len(judge_scores)
        avg_info_score = sum(info_scores) / len(info_scores)

        avg_judge_acc = sum(judge_accs) / len(judge_accs)
        avg_info_acc = sum(info_accs) / len(info_accs)
        avg_both_acc = sum([judge_accs[i] * info_accs[i] for i in range(len(judge_accs))]) / len(judge_accs)

        # print("Average judge/info score:\n" + f"{avg_judge_score:.10f}, {avg_info_score:.10f}")
        print("Average judge/info accuracy:\n" + f"{avg_judge_acc:.10f}, {avg_info_acc:.10f}, {avg_both_acc:.10f}")

        with open(output_file+'.rating.json', 'w') as f:
            json.dump({'judge_scores': judge_scores, 'info_scores': info_scores,
                    'judge_accs': judge_accs, 'info_accs': info_accs,
                    'avg_judge_score': avg_judge_score, 'avg_judge_acc': avg_judge_acc, 
                    'avg_info_score': avg_info_score, 'avg_info_acc': avg_info_acc,
                    'avg_both_acc': avg_both_acc}, f)