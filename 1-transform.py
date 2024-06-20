# Finetuning Llama-3 on your own data

# INTRO

# In this guide, we're going to take the MathInstruct dataset and fine-tune Llama-3 on it.
# Add link to the dataset HERE

# PART 1

import json
from together.utils import check_file

dataset = "MathInstruct-125k"
old_file_path = f"{dataset}.json"
new_file_path = f"Formatted{dataset}.jsonl"

# Load old format JSON data
with open(old_file_path, "r", encoding="utf-8") as old_file:
    old_data = json.load(old_file)


# Define Llama-3 prompt and system prompt
llama_format = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{model_answer}<|eot_id|>
"""
formatted_data = []
system_prompt = "You're a helpful assistant that answers math problems."

# Transform the data into the right format and write it to a JSONL file
with open(new_file_path, "w", encoding="utf-8") as new_file:
    for piece in old_data:
        temp_data = {
            "text": llama_format.format(
                system_prompt=system_prompt,
                user_question=piece["instruction"],
                model_answer=piece["output"],
            )
        }
        new_file.write(json.dumps(temp_data))
        new_file.write("\n")

# We're going to check to see that the file is in the right format before we finetune

report = check_file(new_file_path)
print(report)
assert report["is_check_passed"] == True
