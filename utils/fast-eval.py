# [OPTIONAL] – an evaluation script to test the accuracy of the fine-tuned model vs the base model.
# Benefits from higher rate-limits.

# We install dataformer to make parallel async requests to together api endpoint. Also, supports caching (request_list must be same).
# pip install dataformer@git+https://github.com/DataformerAI/dataformer.git 

import os
import json
from dataformer.llms import AsyncLLM

api_key=os.environ.get("TOGETHER_API_KEY")
max_requests_per_minute = 300

base_model = "meta-llama/Llama-3-8b-chat-hf"
top_oss_model = "meta-llama/Llama-3-70b-chat-hf"

# Replace this with your own model.
finetuned_model = "google/gemma-2-9b-it"

evaluator_model = "meta-llama/Llama-3-70b-chat-hf"
eval_dataset = "EvalDataset-100.json"

if base_model == top_oss_model or base_model == finetuned_model or top_oss_model == finetuned_model:
    raise ValueError("Base model, top OSS model and fine-tuned model must be different.")

llm = AsyncLLM(api_provider="together", max_requests_per_minute=max_requests_per_minute, api_key=api_key)

# 1. Get all responses for the eval dataset
with open(eval_dataset, "r", encoding="utf-8") as eval_file:
    eval_data = json.load(eval_file)

def initial_request_list(model, eval_data):
    return [
        {"model": model, "messages": [{"role": "user", "content": example["instruction"]}], "max_tokens": 1500} for example in eval_data
    ]


request_list = initial_request_list(base_model, eval_data) + initial_request_list(top_oss_model, eval_data) + initial_request_list(finetuned_model, eval_data)
completions = llm.generate(request_list)

# Initialize empty lists for each model's completions
baseModelCompletions = []
topOSSModelCompletions = []
finetunedModelCompletions = []

# Iterate over completions and categorize them based on the model name
for completion in completions:
    model_name = completion[0]['model']
    message_content = completion[1]['choices'][0]['message']['content']
    
    if model_name == base_model:
        baseModelCompletions.append(message_content)
    elif model_name == top_oss_model:
        topOSSModelCompletions.append(message_content)
    elif model_name == finetuned_model:
        finetunedModelCompletions.append(message_content)

results = []

for idx, example in enumerate(eval_data):

    results.append(
        {
            "groundTruthAnswer": example["output"],
            "baseModelAnswer": baseModelCompletions[idx],
            "topOSSModelAnswer": topOSSModelCompletions[idx],
            "fineTunedModelAnswer": finetunedModelCompletions[idx],
        }
    )

# 2. Send the responses from base model & finetuned model to LLama-3-70B to grade them on accuracy
with open("results.json", "w", encoding="utf-8") as results_file:
    json.dump(results, results_file, indent=4)

# Function to prepare the request list
def prepare_request_list(model_answer_key, evaluator_model):
    return [
        {
            "model": evaluator_model,
            "metadata": {"mak": model_answer_key},
            "messages": [
                {
                    "role": "system",
                    "content": "You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INACCURATE otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this.",
                },
                {
                    "role": "user",
                    "content": f"""
                <GroundTruthAnswer>
                {result["groundTruthAnswer"]}
                </GroundTruthAnswer>
                <ModelAnswer>
                {result[model_answer_key]}
                </ModelAnswer>
                """,
                },
            ],
        } for result in results
    ]

request_list = prepare_request_list("baseModelAnswer", evaluator_model) + prepare_request_list("topOSSModelAnswer", evaluator_model) + prepare_request_list("fineTunedModelAnswer", evaluator_model)

evaluations = llm.generate(request_list)

base_model_evaluations = []
top_oss_model_evaluations = []
finetuned_model_evaluations = []

for evaluation in evaluations:
    mak = evaluation[-1]['mak']
    
    if mak == "baseModelAnswer":
        base_model_evaluations.append(evaluation)
    elif mak == "topOSSModelAnswer":
        top_oss_model_evaluations.append(evaluation)
    elif mak == "fineTunedModelAnswer":
        finetuned_model_evaluations.append(evaluation)

badResponses = 0

# Function to count the results
def count_results(evaluations):
    global badResponses
    count = 0
    for evaluation in evaluations:
        eval_answer = evaluation[1]["choices"][0]["message"]["content"]
        if eval_answer== "ACCURATE":
            count += 1
        elif eval_answer != "INACCURATE":
            badResponses += 1
    return count

baseModelCount = count_results(base_model_evaluations)
topOSSModelCount = count_results(top_oss_model_evaluations)
fineTunedModelCount = count_results(finetuned_model_evaluations)

print("Base model count: ", baseModelCount)
print("Top OSS model count: ", topOSSModelCount)
print("Fine-tuned model count: ", fineTunedModelCount)
print("Bad responses count: ", badResponses)

print("\n=== Results – accuracy (%) ===\n")
print("Base model (Llama-3-8b): ", f"{baseModelCount / len(results) *100 }%")
print(
    "Top OSS model (Llama-3-70b): ",
    f"{topOSSModelCount / len(results) * 100}%",
)
print("Fine-tuned model: ", f"{fineTunedModelCount / len(results) *100 }%")