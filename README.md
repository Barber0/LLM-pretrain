# LLM from Scratch: Large Language Model Pretraining and Fine-Tuning Scripts

English | [中文](README-cn.md)

This project provides a set of scripts for pretraining and fine-tuning large language models, enabling you to quickly build your own natural language processing applications or research projects.

## Background

With the advancement of artificial intelligence and natural language processing, large language models such as GPT-4 have become the core of many applications and research projects. Pretraining and fine-tuning are common methods for training these large models. They involve learning the statistical properties of language from large-scale text data and fine-tuning the models on specific tasks to improve performance.

This project aims to simplify the process of pretraining and fine-tuning, providing a set of user-friendly scripts that make it easier for you to work with large language models.

## Features

- Provides pretraining scripts for training language models from large-scale text data.
- Implements a GPT-like large language model using PyTorch.
- Provides distributed pretraining scripts that support training techniques such as Data Parallelism.
- Provides support for FlashAttention to accelerate training.
- Apply RoPE as the position embedding. Implement a simulated complex RoPE based on real FP16, let the scripts and models can run in TPU.

## Getting Started

To get started with this project, follow these steps:

1. Clone this project to your local machine: `git clone https://github.com/Barber0/LLM-pretrain.git`
2. Install the required dependencies: `pip install transformers tokenizers datasets deepspeed`
3. (**Optional**) Install FlashAttention: `pip install flash-attn --no-build-isolation`
4. Run the pretraining script to train the language model: `./scripts/train_with_dp.sh`

Please note that to successfully run the pretraining and fine-tuning scripts, you need to prepare appropriate training datasets and set the corresponding paths and parameters in the configuration file. Several DataLoader options are provided for the following datasets:

1. [EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile)
2. [YeungNLP/ultrachat](https://huggingface.co/datasets/YeungNLP/ultrachat)
3. [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)

## License

This project is licensed under the [MIT License](LICENSE). See the license file for more information.

---

We hope that this project helps you achieve better results in language model pretraining and fine-tuning! If you have any questions or need further assistance, please feel free to contact us.

Wishing you success in your natural language processing journey!

---
