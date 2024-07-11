# Finetuning Llama-3 on your own data

This repo gives you the code to fine-tune Llama-3 on your own data. In this example, we'll be finetuning on 500 pieces of data from the [Math Instruct dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) from TIGER-Lab. LLMs are known for not being the best at complex multi-step math problems so we want to fine-tune an LLM on some of these problems and see how well it does.

We'll go through data cleaning, uploading your dataset, fine-tuning LLama-3-8B on it, then running evals to show the accuracy vs the base model. Fine-tuning will happen on Together and costs $5 with the current pricing.

## Fine-tuning Llama-3 on MathInstruct

1. Make an account at [Together AI](https://www.together.ai/) and save your API key as an OS variable called `TOGETHER_API_KEY`.
2. Install the Together AI python library by running `pip install together`.
3. [Optional] Make an account with Weights and Biases and save your API key as `WANDB_API_KEY`.
4. Run `1-transform.py` to do some data cleaning and get it into a format Together accepts.
5. Run `2-finetune.py` to upload the dataset and start the fine-tuning job on Together.
6. Run `3-eval.py` to evaluate the fine-tuned model against a base model and get accuracy.
7. [Optional] Run `utils/advanced-eval.py` to run the model against other models like GPT-4 as well.

## Results

> Note: This repo contains 500 problems for training but we finetuned our model on 207k problems

After fine-tuning Llama-3-8B on 207k math problems from the MathInstruct dataset, we ran an eval of 1000 new math problems through to compare. Here were the results:

- Base model (Llama-3-8b): 47.2%
- Fine-tuned (Llama-3-8b) model: 65.2%
- Top OSS model (Llama-3-70b): 64.2%
- Top proprietary model (GPT-4o): 71.4%
