import json 
import os 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--file_base', type=str, default='llama7b_eval_baseline')
args = parser.parse_args()

OUT_DIR = "/iris/u/kattian/project_hallucination/DoLa/outputs/bio"
GEN_DIR = "/iris/u/kattian/project_hallucination/DoLa/outputs/bio/for_fseval"
if not os.path.exists(GEN_DIR):
    os.makedirs(GEN_DIR)

file_base = args.file_base
gen_file = os.path.join(GEN_DIR, file_base + '_generations.json')
gen_dict = {}

for split in ["train", "test"]:
    split_file = os.path.join(OUT_DIR, file_base + f'_{split}.json')

    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            dola_out = json.load(f)
        names = dola_out['question']
        generations = dola_out['model_completion']
        prompts = dola_out['prompt']

        gen_dict[split] = {}
        for i in range(len(names)):
            gen_dict[split][names[i]] = {prompts[i]: {"generation": generations[i], "prompt": prompts[i]}}
        
        gen_dict[f'{split}_dola_args'] = dola_out['args']

# save new gen_dict to gen_file 
with open(gen_file, 'w') as f:
    json.dump(gen_dict, f, indent=4)