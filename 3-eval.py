# PART 3

# An evaluation script to test the accuracy of the fine-tuned model vs the base model.

import json
from typing import List
from pydantic import BaseModel
from together import Together
import os


class EvalData(BaseModel):
    instruction: str
    output: str


class EvalDataset:
    data: List[EvalData]


client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

with open("EvalDataset-2.json", "r", encoding="utf-8") as eval_file:
    eval_data: EvalDataset = json.load(eval_file)

# print(eval_data)

results = []

with open("answer.json", "r", encoding="utf-8") as eval_file:
    results = json.load(eval_file)

# for example in eval_data:
#     baseModelCompletion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You're a helpful medical doctor who answers questions.",
#             },
#             {"role": "user", "content": example["instruction"]},
#         ],
#         model="meta-llama/Llama-3-8b-chat-hf",
#     )

#     finetunedModelCompletion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You're a helpful medical doctor who answers questions.",
#             },
#             {"role": "user", "content": example["instruction"]},
#         ],
#         model="hassan@together.ai/Meta-Llama-3-8B-Instruct-mathinstruct-125k-wandb-2024-06-19-20-47-17-4b51b635",
#     )

#     results.append(
#         {
#             "groundTruthAnswer": example["output"],
#             "baseModelAnswer": baseModelCompletion.choices[0].message.content,
#             "fineTunedModelAnswer": finetunedModelCompletion.choices[0].message.content,
#         }
#     )


#         {
#             "groundTruthAnswer": example["output"],
#             "baseModelAnswer": baseModelCompletion.choices[0].message.content,
# "baseModelAnswerAccurate": false,
#             "fineTunedModelAnswer": finetunedModelCompletion.choices[0].message.content,
# "fineTunedModelAnswerAccurate": true,
#         }

# baseModelCount = results.reduce((acc, curr) => curr.baseModelAnswerAccurate ? acc + 1 : acc, 0)
# 100
# fineTunedModelCount = results.reduce((acc, curr) => curr.fineTunedModelAnswerAccurate ? acc + 1 : acc, 0)
# 150

# baseModel accuracy: count / total = 100 / 150 = 66%
# fineTuned accuracy: count / total = 140 / 150 = 90%

# print(results)

# send both results to LLama-3-70B to determine if it got it right or wrong

baseModelCount = 0
fineTunedModelCount = 0

for result in results:
    baseModelAnswer = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INACCURATE otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this.",
            },
            {
                "role": "user",
                "content": f"""
              <GroundTruthAnswer> {result["groundTruthAnswer"]} </GroundTruthAnswer>
              <ModelAnswer> {result["baseModelAnswer"]} </ModelAnswer>
              """,
            },
        ],
        model="meta-llama/Llama-3-70b-chat-hf",
    )
    if baseModelAnswer.choices[0].message.content == "ACCURATE":
        baseModelCount += 1

    fineTunedModelAnswer = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INACCURATE otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this.",
            },
            {
                "role": "user",
                "content": f"""
              <GroundTruthAnswer> {result["groundTruthAnswer"]} </GroundTruthAnswer>
              <ModelAnswer> {result["fineTunedModelAnswer"]} </ModelAnswer>
              """,
            },
        ],
        model="meta-llama/Llama-3-70b-chat-hf",
    )
    if fineTunedModelAnswer.choices[0].message.content == "ACCURATE":
        fineTunedModelCount += 1

print("Base model count: ", baseModelCount)
print("Fine-tuned model count: ", fineTunedModelCount)
