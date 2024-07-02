# [OPTIONAL] PART 4 – an evaluation script to test the accuracy of the fine-tuned model vs the base model and other state of the art models.

import json
from together import Together, AsyncTogether
from openai import AsyncOpenAI
import os
import asyncio

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_together_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

base_model = "meta-llama/Llama-3-8b-chat-hf"
top_oss_model = "meta-llama/Llama-3-70b-chat-hf"
top_gpt_model = "gpt-4o"
finetuned_model = "hassan@together.ai/Meta-Llama-3-8B-Instruct-mathinstruct-125k-v2-2024-06-20-17-54-50-f40b62b6"
evaluator_model = "meta-llama/Llama-3-70b-chat-hf"


async def chatCompletion(model, instruction):
    completion = await async_together_client.chat.completions.create(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model=model,
        max_tokens=1500,
    )
    print(model + "is done")
    return completion.choices[0].message.content


async def openAICompletion(model, instruction):
    completion = await async_openai_client.chat.completions.create(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model="gpt-4o",
    )
    print(model + "is done")
    return completion.choices[0].message.content


async def main():
    # 1. Get all responses for the eval dataset
    with open("EvalDataset-1000.json", "r", encoding="utf-8") as eval_file:
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

    async def evalCompletion(count, modelAnswer):
        isAccurate = await client.chat.completions.create(
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
                        {modelAnswer}
                        </ModelAnswer>
                        """,
                },
            ],
            model=evaluator_model,
        )
        if isAccurate.choices[0].message.content == "ACCURATE":
            count += 1

        if (
            isAccurate.choices[0].message.content != "INACCURATE"
            and isAccurate.choices[0].message.content != "ACCURATE"
        ):
            badResponses += 1

    for result in results:
        try:
            await asyncio.gather(
                evalCompletion(baseModelCount, result["baseModelAnswer"]),
                evalCompletion(topOSSModelCount, result["topOSSModelAnswer"]),
                evalCompletion(gptModelCount, result["topGptAnswer"]),
                evalCompletion(fineTunedModelCount, result["fineTunedModelAnswer"]),
            )
        except Exception:
            numErrors += 1
            print("Error in response: ", Exception)

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


asyncio.run(main())
