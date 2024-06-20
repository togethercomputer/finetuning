# PART 3

# An evaluation script to test the accuracy of the fine-tuned model vs the base model.

import json
from together import Together
import os

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

base_model = "meta-llama/Llama-3-8b-chat-hf"
finetuned_model = "hassan@together.ai/Meta-Llama-3-8B-Instruct-mathinstruct-125k-wandb-2024-06-19-20-47-17-4b51b635"
evaluator_model = "meta-llama/Llama-3-70b-chat-hf"

# 1. Get all responses for the eval dataset
with open("EvalDataset-100.json", "r", encoding="utf-8") as eval_file:
    eval_data = json.load(eval_file)

results = []

for example in eval_data:
    baseModelCompletion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You're a helpful medical doctor who answers questions.",
            },
            {"role": "user", "content": example["instruction"]},
        ],
        model=base_model,
        max_tokens=2000,
    )

    finetunedModelCompletion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You're a helpful medical doctor who answers questions.",
            },
            {"role": "user", "content": example["instruction"]},
        ],
        model=finetuned_model,
        max_tokens=2000,
    )

    results.append(
        {
            "groundTruthAnswer": example["output"],
            "baseModelAnswer": baseModelCompletion.choices[0].message.content,
            "fineTunedModelAnswer": finetunedModelCompletion.choices[0].message.content,
        }
    )

# 2. Send the responses from base model & finetuned model to LLama-3-70B to grade them on accuracy
with open("results.json", "w", encoding="utf-8") as results_file:
    json.dump(results, results_file, indent=4)

baseModelCount = 0
fineTunedModelCount = 0
badResponses = 0

for result in results:
    try:
        baseModelAnswer = client.chat.completions.create(
            messages=[
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
                  {result["baseModelAnswer"]}
                  </ModelAnswer>
                  """,
                },
            ],
            model=evaluator_model,
        )
        if baseModelAnswer.choices[0].message.content == "ACCURATE":
            baseModelCount += 1

        if (
            baseModelAnswer.choices[0].message.content != "INACCURATE"
            and baseModelAnswer.choices[0].message.content != "ACCURATE"
        ):
            badResponses += 1

        fineTunedModelAnswer = client.chat.completions.create(
            messages=[
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
                  {result["fineTunedModelAnswer"]}
                  </ModelAnswer>
                  """,
                },
            ],
            model=evaluator_model,
        )
        if fineTunedModelAnswer.choices[0].message.content == "ACCURATE":
            fineTunedModelCount += 1
        if (
            fineTunedModelAnswer.choices[0].message.content != "INACCURATE"
            and fineTunedModelAnswer.choices[0].message.content != "ACCURATE"
        ):
            badResponses += 1
    except Exception:
        print("Error in response: ", Exception)

print("Base model count: ", baseModelCount)
print("Fine-tuned model count: ", fineTunedModelCount)
print("Bad responses count: ", badResponses)

print("\n=== Results ===\n")
print("Base model accuracy: ", f"{baseModelCount / len(results) * 100}%")
print("Fine-tuned model accuracy: ", f"{fineTunedModelCount / len(results) * 100}%")
