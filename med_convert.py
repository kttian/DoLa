## conversion for medqa

import json 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_base', type=str, default='llama7b_eval_dola_v0_stopword')
parser.add_argument('--dataset', type=str, default='medqa')
args = parser.parse_args()

OUT_DIR = f"/iris/u/kattian/project_hallucination/DoLa/outputs/{args.dataset}"
GEN_DIR = f"/iris/u/kattian/project_hallucination/DoLa/outputs/{args.dataset}/for_fseval"
if not os.path.exists(GEN_DIR):
    os.makedirs(GEN_DIR)

file_base = args.file_base
# test_file = file_base + '_test.json'
# train_file = file_base + '_train.json'
# gen_file = 'for_fseval/' + file_base + '_generations.json'

# filenames = {'train': train_file, 'test': test_file}

gen_file = os.path.join(GEN_DIR, file_base + '_generations.json')
new_dict = {}

for split in ["train", "test"]:
    file = os.path.join(OUT_DIR, file_base + f'_{split}.json')

    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        new_dict[split] = {}
        names = data['name']
        questions = data['question']
        generations = data['model_completion']

        for i in range(len(names)):
            name = names[i]
            question = questions[i]
            generation = generations[i].split("\nQUESTION:")[0].strip()
            if name not in new_dict[split]:
                new_dict[split][name] = {}
            new_dict[split][name][question] = {'generation': generation}
        new_dict[f'{split}_dola_args'] = data['args']
    # else:
    #     new_dict[split] = {}

# save new_dict to gen_file 
print("output file:", gen_file)
with open(gen_file, 'w') as f:
    json.dump(new_dict, f, indent=4)
