# Finetuning Llama-3 on your own data

This repo helps you fine-tune Llama-3 on your own data. In this example, we'll be finetuning on 125k samples of the Math dataset from huggingface.

## Running the repo

1. Make an account at [Together AI](https://www.together.ai/) and save your API key as an OS variable called `TOGETHER_API_KEY`
2. [Optional] Make an account with Weights and Biases and save your API key as `WANDB_API_KEY`
3. Run `1-transform.py` to do some data cleaning and get it into a format Together accepts
4. Run `2-finetune.py` to upload the dataset and start the fine-tuning job on Together
5. Run `3-eval.py` to evaluate the fine-tuned model against a base model and get accuracy
