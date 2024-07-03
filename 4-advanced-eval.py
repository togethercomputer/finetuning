# [OPTIONAL] PART 4 – an evaluation script to test the accuracy of the fine-tuned model vs the base model and other state of the art models.

import json
from together import Together, AsyncTogether
from openai import AsyncOpenAI
import os
import asyncio
import time

async_together_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

base_model = "meta-llama/Llama-3-8b-chat-hf"
top_oss_model = "meta-llama/Llama-3-70b-chat-hf"
top_gpt_model = "gpt-4o"
finetuned_model = "hassan@together.ai/Meta-Llama-3-8B-Instruct-mathinstruct-125k-v2-2024-06-20-17-54-50-f40b62b6"
evaluator_model = "meta-llama/Llama-3-70b-chat-hf"
eval_dataset = "EvalDataset-5.json"

async def chatCompletion(model, instruction):
    completion = await async_together_client.chat.completions.create(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model=model,
        max_tokens=1500,
    )
    return completion.choices[0].message.content


async def openAICompletion(model, instruction):
    completion = await async_openai_client.chat.completions.create(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model="gpt-4o",
    )
    return completion.choices[0].message.content


async def main():
    start_time = time.time()  # Start timing

    # 1. Get all responses for the eval dataset
    with open(eval_dataset, "r", encoding="utf-8") as eval_file:
        eval_data = json.load(eval_file)

    results = []

    for example in eval_data:
        (
            baseModelCompletion,
            topOSSModelCompletion,
            topGptCompletion,
            finetunedModelCompletion,
        ) = await asyncio.gather(
            chatCompletion(base_model, example["instruction"]),
            chatCompletion(top_oss_model, example["instruction"]),
            openAICompletion(top_gpt_model, example["instruction"]),
            chatCompletion(finetuned_model, example["instruction"]),
        )
        results.append(
            {
                "groundTruthAnswer": example["output"],
                "baseModelAnswer": baseModelCompletion,
                "topOSSModelAnswer": topOSSModelCompletion,
                "topGptAnswer": topGptCompletion,
                "fineTunedModelAnswer": finetunedModelCompletion,
            }
        )

    # 2. Send the responses from base model & finetuned model to LLama-3-70B to grade them on accuracy
    with open("results.json", "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=4)

    baseModelCount = 0
    topOSSModelCount = 0
    gptModelCount = 0
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
            baseModelCount_inc, topOSSModelCount_inc, gptModelCount_inc, fineTunedModelCount_inc = await asyncio.gather(
                evalCompletion(result["groundTruthAnswer"], result["baseModelAnswer"]),
                evalCompletion(result["groundTruthAnswer"], result["topOSSModelAnswer"]),
                evalCompletion(result["groundTruthAnswer"], result["topGptAnswer"]),
                evalCompletion(result["groundTruthAnswer"], result["fineTunedModelAnswer"])
            )

            baseModelCount += baseModelCount_inc[0]
            topOSSModelCount += topOSSModelCount_inc[0]
            gptModelCount += gptModelCount_inc[0]
            fineTunedModelCount += fineTunedModelCount_inc[0]

            badResponses += (baseModelCount_inc[1] + topOSSModelCount_inc[1] +
                             gptModelCount_inc[1] + fineTunedModelCount_inc[1])
        except Exception as e:
            numErrors += 1
            print("Error in response: ", e)

    print("Base model count: ", baseModelCount)
    print("Top OSS model count: ", topOSSModelCount)
    print("GPT-4o count: ", gptModelCount)
    print("Fine-tuned model count: ", fineTunedModelCount)
    print("Bad responses count: ", badResponses)
    print("Number of errors: ", numErrors)

    print("\n=== Results – accuracy (%) ===\n")
    print("Base model (Llama-3-8b): ", f"{baseModelCount / len(results) * 100}%")
    print(
        "Top OSS model (Llama-3-70b): ",
        f"{topOSSModelCount / len(results) * 100}%",
    )
    print("GPT-4o: ", f"{gptModelCount / len(results) * 100}%")
    print("Fine-tuned model: ", f"{fineTunedModelCount / len(results) * 100}%")

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

asyncio.run(main())
