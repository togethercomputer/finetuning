# Finetuning Llama-3 on your own data

This repo gives you the code to fine-tune Llama-3 on your own data. In this example, we'll be finetuning on 125k pieces of data from the [Math Instruct dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) from TIGER-Lab. LLMs are known for not being the best at complex multi-step math problems so we want to fine-tune an LLM on 125k of these problems and see how well it does.

We'll go through data cleaning, uploading your dataset, fine-tuning LLama-3-8B on it, then running evals to show the accuracy vs the base model. Fine-tuning will happen on Together and costs ~$35 with the current pricing.

## Fine-tuning Llama-3 on MathInstruct

1. Make an account at [Together AI](https://www.together.ai/) and save your API key as an OS variable called `TOGETHER_API_KEY`
2. [Optional] Make an account with Weights and Biases and save your API key as `WANDB_API_KEY`
3. Run `1-transform.py` to do some data cleaning and get it into a format Together accepts
4. Run `2-finetune.py` to upload the dataset and start the fine-tuning job on Together
5. Run `3-eval.py` to evaluate the fine-tuned model against a base model and get accuracy

## Results

After fine-tuning Llama-3-8B on 125k math problems from the MathInstruct dataset, we ran 100 new math problems through to compare. The fine-tuned model was 79% accurate, up from 46% accuracy of the base model. Also worth noting that the fine-tuned model produces results that were more succint & better overall.

Link to full blog post:
