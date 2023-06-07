import os
import json
import subprocess
import ast

import tqdm
from evalplus.data import get_human_eval_plus, write_jsonl

from evaluation.utils import replace_model_name_slashes, create_model

def postprocess_model_reply(model_reply):
    model_reply = model_reply.split('```python')[-1].split('```py')[-1].split('```Python')[-1]
    if model_reply.count('```') >= 2:
        model_reply =  model_reply.split('```')[-1]
    model_reply = model_reply.split('```')[0]
    model_reply = model_reply.strip('\n')
    return model_reply

def evaluate_model(model_type, model_name):
    output_path = os.path.join('./reports/human-eval', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(output_path):
        return

    model = create_model(model_type, model_name)

    dataset = get_human_eval_plus()
    samples = []
    for task_id in tqdm.tqdm(dataset):
        prompt = dataset[task_id]['prompt']
        reply = model.reply([
            ('user',
                'Please complete the following Python code. '
                'Do NOT provide any additional explanation or tests. '
                'Also do NOT output things like "Sure!" or "Here you go!" or similar things. '
                'Just provide the code without anything else.'
                '\n\n'
                + prompt),
        ])

        reply = postprocess_model_reply(reply)
        print('@@@@@@@@@@@@@@@@@@@@\n' + reply + '\n@@@@@@@@@@@@@@@@@@@@')
        samples.append({ 'task_id': task_id, 'completion': reply })

    write_jsonl('human-eval-plus-tmp.jsonl', samples)
    process_output = subprocess.run([
        'evalplus.evaluate',
        '--dataset', 'humaneval',
        '--samples', 'human-eval-plus-tmp.jsonl'
    ], capture_output=True, text=True).stdout
    os.remove('human-eval-plus-tmp.jsonl')

    with open('human-eval-plus-tmp_eval_results.json') as f:
        results = json.load(f)['eval']

    samples_with_results = []
    for sample in samples:
        task_id = sample['task_id']
        completion = sample['completion']
        prompt = dataset[task_id]['prompt']
        result = results[task_id]['plus'][0][0]
        if result == 'failed':
            success = False
        elif result == 'success':
            success = True
        else:
            print(task_id, results[task_id]['plus'][0][0])
            raise
        samples_with_results.append({
            'task_id': task_id,
            'prompt': prompt,
            'completion': completion,
            'success': success,
        })

    os.remove('human-eval-plus-tmp_eval_results.json')

    process_output_lines = [line for line in process_output.split('\n') if line != '']
    assert process_output_lines[-2] == 'Base + Extra'
    score = ast.literal_eval(process_output_lines[-1])['pass@1']
    output = { 'replies': samples_with_results, 'score': score }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
