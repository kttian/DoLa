import json 
import os 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--file_base', type=str, default='llama7b_eval_baseline')
args = parser.parse_args()

file_base = args.file_base
test_file = file_base + '_test.json'
train_file = file_base + '_train.json'
gen_file = 'for_fseval/' + file_base + '_generations400.json'

new_dict = {}
if os.path.exists(test_file):
    with open(test_file, 'r') as f:
        data = json.load(f)
    new_dict['test_names'] = data['question']
    new_dict['test_generations'] = data['model_completion']
else:
    new_dict['test_names'] = []
    new_dict['test_generations'] = []

if os.path.exists(train_file):
    with open(train_file, 'r') as f:
        data = json.load(f)
    new_dict['train_names'] = data['question']
    new_dict['train_generations'] = data['model_completion']
else:
    new_dict['train_names'] = []
    new_dict['train_generations'] = []

# save new_dict to gen_file 
with open(gen_file, 'w') as f:
    json.dump(new_dict, f)