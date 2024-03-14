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

import ssl
import urllib.request
import zipfile
import time 

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

def get_questions_dict():
    questions_file = "/iris/u/kattian/project_hallucination/sft_rl_0719/generate_lfqa_medical_data/all_questions_simple.json"
    with open(questions_file, 'r') as f:
        questions_dict = json.load(f)
    return questions_dict 

def get_wiki_list(split=None):
    wiki_file = "/iris/u/kattian/project_hallucination/sft_rl_0719/generate_lfqa_medical_data/wiki_list.txt"
    with open(wiki_file, 'r') as f:
        wiki_list = f.read().split("\n")
    if split is None:
        return wiki_list
    if split == "train":
        return wiki_list[:150]
    if split == "test":
        return wiki_list[150:200]
    else:
        return wiki_list 

def build_prompt(question):
    EXAMPLE_Q1 = "I know there are different types of strokes. Can you explain the differences and their causes?"
    EXAMPLE_A1 = "There are two major types of strokes: ischemic and hemorrhagic. Ischemic strokes occur when blood vessels supplying the brain become blocked by blood clots. Hemorrhagic strokes are caused by leaks or bleeding in the blood vessels of the brain. The most common cause of ischemic stroke is atherosclerosis (hardening of the arteries), which leads to narrowing of the arteries in the brain. Blood clots can also form in areas of atherosclerosis and then block the artery completely. The most common cause of hemorrhagic stroke is the rupture of a small artery in the brain, leading to bleeding. This is called a cerebral hemorrhage."

    EXAMPLE_Q2 = "How does treating frostbite differ depending on the severity?"
    EXAMPLE_A2 = "Depending on the extent of frostbite, a medical professional may recommend treatment in one of three ways. For mild frostbite, applying warm cloths to affected areas or applying a warm water bottle against the skin are common treatments. However, moderate frostbite may require more extensive treatment, including the use of warm water or an electric blanket. Severe frostbite often requires surgery in order to avoid skin loss. The frostbitten area is initially treated with warm water and may be wrapped in a moist bandage. This is followed by a process called debridement, which involves surgically removing damaged tissue."

    EXAMPLE_Q3 = "What are the symptoms of kidney stone disease?"
    EXAMPLE_A3 = "Symptoms of kidney stones vary depending on where the stone is located and the size of the stone. Kidney stones may be present in the kidney and remain undetected unless they begin to move to the ureter (tube that carries urine from the kidney to the bladder). Then, symptoms that usually occur include sharp pain in the back or side that radiates to the groin or genital area, and blood in the urine. The pain can be quite severe, and people often compare the pain to labor pains or the feeling of getting a kick in the abdomen. Often, the pain will get worse when urinating."

    EXAMPLE_Q4 = "What is the recommended treatment for appendicitis?" 
    EXAMPLE_A4 = "Treatment will depend on whether the appendix has burst and how sick the patient is. The typical treatment for appendicitis includes antibiotics, a pain reliever, and, if needed, surgery. There are several different surgical options to treat appendicitis, but the most common one is appendectomy, or removal of the appendix. This can be performed either through an open incision or laparoscopically. Surgical complications include bleeding, infection and wound healing issues, but most patients do well after surgery."

    EXAMPLE_Q5 = "Can you explain how gastroenteritis is typically transmitted?"
    EXAMPLE_A5 = "Gastroenteritis, which refers to inflammation of the digestive tract, is transmitted when an individual ingests food or water that is contaminated with the bacteria, virus or parasite that causes gastroenteritis. This type of illness can be spread through close personal contact with someone who has the infection, and touching items or surfaces that are contaminated with feces."

    EXAMPLE_Q6 = "What are the potential complications if edema is left untreated and how can I prevent them?"
    EXAMPLE_A6 = "Edema can be dangerous if left untreated, as it can lead to serious complications like congestive heart failure, cardiac tamponade and other conditions that can be life-threatening. People who suffer from edema should have a health care provider monitor their heart, as this can be affected by fluid build-up. Individuals can prevent edema by staying well hydrated, making sure to get regular exercise and keeping a healthy diet with low sodium. They should also get the medical professional's approval before taking over-the-counter medications to treat this condition."

    prompt = ""
    prompt += f"QUESTION: {EXAMPLE_Q1} ANSWER: {EXAMPLE_A1}\n"
    prompt += f"QUESTION: {EXAMPLE_Q2} ANSWER: {EXAMPLE_A2}\n"
    prompt += f"QUESTION: {EXAMPLE_Q3} ANSWER: {EXAMPLE_A3}\n"
    prompt += f"QUESTION: {EXAMPLE_Q5} ANSWER: {EXAMPLE_A5}\n"
    prompt += f"QUESTION: {EXAMPLE_Q6} ANSWER: {EXAMPLE_A6}\n"
    prompt += f"QUESTION: {EXAMPLE_Q4} ANSWER: {EXAMPLE_A4}\n"
    prompt += f"QUESTION: {question} ANSWER:"
    return prompt 


# def build_prompt(input_name):
#     demo = get_few_shot_prompt()
#     input_text_prompt = demo + "NAME: " + input_name + ". BIOGRAPHY:"
#     return input_text_prompt

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
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=None) # changed from None to 1.2
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
    questions_dict = get_questions_dict()
    names = get_wiki_list(split=args.split)[:args.size]
    list_data_dict = []
    list_data_dict_names = []
    for name in names:
        list_data_dict.extend(questions_dict[name]['simple_questions'])
        list_data_dict_names.extend([name] * len(questions_dict[name]['simple_questions']))
    
    if args.debug:
        list_data_dict = list_data_dict[:10]

    print("list data dict:", list_data_dict)
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory, args.checkpoint_path)
    stop_word_list = ["\nQUESTION:"]
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
    answers = []
    result_dict = {'question': [], 'model_completion': [], 'name': [], 'prompt': []}
    i = -1
    for sample in tqdm(list_data_dict):
        i += 1

        if args.checkpoint_path: # if load sft model, no few shot prompt
            input_text = sample 
        else:
            input_text = build_prompt(sample)
        result_dict['prompt'].append(input_text)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, 
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
        result_dict['name'].append(list_data_dict_names[i])
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample}\n\n'
            f'Model Completion: {model_completion}\n\n')

        print(f'Num of total question: {len(answers)}.')
    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    result_dict['args'] = vars(args)
    print(f"Total time: {time.time() - start_time:.3f} seconds")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

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