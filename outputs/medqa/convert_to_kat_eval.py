## conversion for medqa

import json 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_base', type=str, default='llama7b_eval_dola_v0_stopword')
args = parser.parse_args()
file_base = args.file_base

test_file = file_base + '_test.json'
train_file = file_base + '_train.json'
gen_file = 'for_fseval/' + file_base + '_generations.json'

filenames = {'train': train_file, 'test': test_file}
new_dict = {}

for split in filenames:
    file = filenames[split]
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
    else:
        new_dict[split] = {}

# save new_dict to gen_file 
with open(gen_file, 'w') as f:
    json.dump(new_dict, f)
