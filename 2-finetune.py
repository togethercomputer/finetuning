# PART 2 – Now that we have the formatted data, we will upload the file to Together AI to be finetuned

import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
dataset = "MathInstruct-500"

# Upload your formatted data and get back the file ID
response = client.files.upload(file=f"Formatted{dataset}.jsonl")
fileId = response.model_dump()["id"]

# Check to see if the file has uploaded successfully
file_metadata = client.files.retrieve(fileId)
print(file_metadata)

# Trigger fine-tuning job
resp = client.fine_tuning.create(
    suffix="mathinstruct-500",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    training_file=fileId,
    n_epochs=3,
    batch_size=8,
    learning_rate=1e-5,
    wandb_api_key=os.environ.get("WANDB_API_KEY"),
)

print(resp)
