import json
import os

from evaluation.benchmarks.utils import model_name_to_filename
from evaluation.models.models import compute_model_replies, create_model

async def generate_model_outputs(model_type, model_name, model_args, evaluation_id):
    with open("data/creative_writing_prompts_and_criteria.json") as f:
        prompts_and_criteria = json.load(f)

    model = await create_model(model_type, model_name, model_args)

    conversations = [
        {
            "conversation": [("user", prompt_data["writing_prompt"])],
            "temperature": 0.7,
        }
        for prompt_data in prompts_and_criteria.values()
    ]

    model_outputs = await compute_model_replies(
        model,
        conversations,
        progress_bar_description=model_name + " :: Creative Writing :: Generating model outputs",
    )

    output_folder = os.path.join(
        "reports/creative_writing", model_name_to_filename(model_name), evaluation_id
    )
    os.makedirs(output_folder, exist_ok=True)

    model_outputs_dict = {
        prompt_id: model_outputs[i] for i, prompt_id in enumerate(prompts_and_criteria.keys())
    }

    with open(os.path.join(output_folder, "model_outputs.json"), "w") as f:
        json.dump(model_outputs_dict, f, indent=4)

def generate_judge_prompts(model_outputs, reference_responses, writing_prompts):
    judge_prompts = []

    for prompt_id, model_output in model_outputs.items():
        prompt = writing_prompts[prompt_id]
        reference_response = reference_responses[prompt_id]

        for criteria_group in prompt["judging_criteria"]:
            judge_prompt = f"You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-10 scale.\n\n"
            judge_prompt += f"[PROMPT START]\n{prompt['writing_prompt']}\n[PROMPT END]\n\n"
            judge_prompt += f"[REFERENCE RESPONSE (DO NOT JUDGE)]\n{reference_response}\n[REFERENCE RESPONSE END]\n\n"
            judge_prompt += f"[TEST MODEL RESPONSE]\n{model_output}\n[TEST MODEL RESPONSE END]\n\n"
            judge_prompt += f"[Task]\n\nYou are an expert in assessing creative writing. Your task is to score the quality of the test model's response above, by several metrics, on a 0-10 scale.\n\n"
            judge_prompt += "Scoring notes:\n\n"
            judge_prompt += "- You are not scoring the quality of the prompt or the reference response, only the test model response.\n\n"
            judge_prompt += "- The reference model response is to be considered high quality output as a reference point.\n\n"
            judge_prompt += "- Scores of 0 or 10 should not be considered highly unlikely just because they are the max/min. Use the full scoring range as appropriate.\n\n"
            judge_prompt += "- Higher scores do not always indicate better writing; e.g. for metrics like \"Trite\".\n\n"
            judge_prompt += "- If no character bios were specified, the Adherence to Character Bios metric should be 5.\n\n"
            judge_prompt += "- Do not add any commentary or explanation.\n\n"
            judge_prompt += "- In the output, write the metric names exactly as below so they can be parsed.\n\n"
            judge_prompt += "- Do not be biased in favour of overly long output.\n\n"
            judge_prompt += "- You are to write a comprehensive analysis for each of the metrics, then give your scores.\n\n"
            judge_prompt += "- Output format is:\n\n"
            judge_prompt += "[Analysis]\n\nWrite your detailed analysis.\n\n"
            judge_prompt += "[Scores]\n\nMetric 1 name: Score [0-10]\n\nMetric 2 name: ...\n\n"
            judge_prompt += "---\n\n"
            judge_prompt += f"{criteria_group['prefix_text']}\n\n"
            judge_prompt += "\n".join(criteria_group["criteria"])
            judge_prompt += "\n\n--"

            judge_prompts.append(judge_prompt)

    return judge_prompts


async def judge_model_outputs(judge_prompts, judge_model_type, judge_model_name, judge_model_args):
    judge_model = await create_model(judge_model_type, judge_model_name, judge_model_args)

    judge_responses = await compute_model_replies(
        judge_model,
        [{"conversation": [("user", prompt)], "temperature": 0.0} for prompt in judge_prompts],
        progress_bar_description=judge_model_name + " :: Creative Writing :: Judging model outputs",
    )

    scores = []
    for response in judge_responses:
        analysis, scores_text = response.split("[Analysis]")[1].split("[Scores]")
        scores_dict = {}
        for line in scores_text.strip().split("\n"):
            if ":" in line:
                metric, score = line.split(":")
                scores_dict[metric.strip()] = float(score.strip())
        scores.append({"analysis": analysis.strip(), "scores": scores_dict})

    return scores

import statistics

def compute_final_score(scores):
    all_scores = []
    for score_dict in scores:
        all_scores.extend(score_dict["scores"].values())

    final_score = statistics.mean(all_scores)

    return final_score

async def evaluate_model(model_type, model_name, model_args, evaluation_id):
    with open("data/creative_writing_prompts_and_criteria.json") as f:
        writing_prompts = json.load(f)

    with open("data/creative_writing_reference_responses.json") as f:
        reference_responses = json.load(f)

    model_outputs_file = os.path.join(
        "reports/creative_writing", model_name_to_filename(model_name), evaluation_id, "model_outputs.json"
    )
    if not os.path.exists(model_outputs_file):
        await generate_model_outputs(model_type, model_name, model_args, evaluation_id)

    with open(model_outputs_file) as f:
        model_outputs = json.load(f)

    judge_prompts = generate_judge_prompts(model_outputs, reference_responses, writing_prompts)

    scores = await judge_model_outputs(
        judge_prompts, "openai", "gpt-4", {"max_new_tokens": 2048}
    )

    final_score = compute_final_score(scores)

    output_folder = os.path.join(
        "reports/creative_writing", model_name_to_filename(model_name), evaluation_id
    )
    with open(os.path.join(output_folder, "scores.json"), "w") as f:
        json.dump({"scores": scores, "final_score": final_score}, f, indent=4)