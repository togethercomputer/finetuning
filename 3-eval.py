# PART 3 – an evaluation script to test the accuracy of the fine-tuned model vs the base model.

import json
from together import AsyncTogether
import os
import asyncio

async_together_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

base_model = "meta-llama/Llama-3-8b-chat-hf"
top_oss_model = "meta-llama/Llama-3-70b-chat-hf"
finetuned_model = "YOUR_FINETUNED_MODEL_ID"
evaluator_model = "meta-llama/Llama-3-70b-chat-hf"
eval_dataset = "EvalDataset-100.json"


async def chatCompletion(model, instruction):
    completion = await async_together_client.chat.completions.create(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model=model,
        max_tokens=1500,
    )
    return completion.choices[0].message.content


async def main():
    # 1. Get all responses for the eval dataset
    with open(eval_dataset, "r", encoding="utf-8") as eval_file:
        eval_data = json.load(eval_file)

    results = []

    for example in eval_data:
        (
            baseModelCompletion,
            topOSSModelCompletion,
            finetunedModelCompletion,
        ) = await asyncio.gather(
            chatCompletion(base_model, example["instruction"]),
            chatCompletion(top_oss_model, example["instruction"]),
            chatCompletion(finetuned_model, example["instruction"]),
        )
        results.append(
            {
                "groundTruthAnswer": example["output"],
                "baseModelAnswer": baseModelCompletion,
                "topOSSModelAnswer": topOSSModelCompletion,
                "fineTunedModelAnswer": finetunedModelCompletion,
            }
        )

    # 2. Send the responses from base model & finetuned model to LLama-3-70B to grade them on accuracy
    with open("results.json", "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=4)

    baseModelCount = 0
    topOSSModelCount = 0
    fineTunedModelCount = 0
    badResponses = 0
    numErrors = 0

    async def evalCompletion(groundTruthAnswer, modelAnswer):
        isAccurate = await async_together_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INACCURATE otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this.",
                },
                {
                    "role": "user",
                    "content": f"""
                        <GroundTruthAnswer>
                        {groundTruthAnswer}
                        </GroundTruthAnswer>

                        <ModelAnswer>
                        {modelAnswer}
                        </ModelAnswer>
                        """,
                },
            ],
            model=evaluator_model,
        )
        if isAccurate.choices[0].message.content == "ACCURATE":
            return 1, 0
        elif isAccurate.choices[0].message.content == "INACCURATE":
            return 0, 0
        else:
            return 0, 1  # Return 1 for badResponses

    for result in results:
        try:
            (
                baseModelCount_inc,
                topOSSModelCount_inc,
                fineTunedModelCount_inc,
            ) = await asyncio.gather(
                evalCompletion(result["groundTruthAnswer"], result["baseModelAnswer"]),
                evalCompletion(
                    result["groundTruthAnswer"], result["topOSSModelAnswer"]
                ),
                evalCompletion(
                    result["groundTruthAnswer"], result["fineTunedModelAnswer"]
                ),
            )

            baseModelCount += baseModelCount_inc[0]
            topOSSModelCount += topOSSModelCount_inc[0]
            fineTunedModelCount += fineTunedModelCount_inc[0]

            badResponses += (
                baseModelCount_inc[1]
                + topOSSModelCount_inc[1]
                + fineTunedModelCount_inc[1]
            )
        except Exception as e:
            numErrors += 1
            print("Error in response: ", e)

    print("Base model count: ", baseModelCount)
    print("Top OSS model count: ", topOSSModelCount)
    print("Fine-tuned model count: ", fineTunedModelCount)
    print("Bad responses count: ", badResponses)
    print("Number of errors: ", numErrors)

    print("\n=== Results – accuracy (%) ===\n")
    print("Base model (Llama-3-8b): ", f"{baseModelCount / len(results) * 100}%")
    print(
        "Top OSS model (Llama-3-70b): ",
        f"{topOSSModelCount / len(results) * 100}%",
    )
    print("Fine-tuned model: ", f"{fineTunedModelCount / len(results) * 100}%")


asyncio.run(main())
