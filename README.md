# LLM from Scratch: Large Language Model Pretraining and Fine-Tuning Scripts

English | [中文](README-cn.md)

This project provides a set of scripts for pretraining and fine-tuning large language models, enabling you to quickly build your own natural language processing applications or research projects.

## Background

With the advancement of artificial intelligence and natural language processing, large language models such as GPT-4 have become the core of many applications and research projects. Pretraining and fine-tuning are common methods for training these large models. They involve learning the statistical properties of language from large-scale text data and fine-tuning the models on specific tasks to improve performance.

This project aims to simplify the process of pretraining and fine-tuning, providing a set of user-friendly scripts that make it easier for you to work with large language models.

## Features

- Provides pretraining scripts for training language models from large-scale text data.
- Implements a GPT-like large language model using PyTorch.
- Provides distributed pretraining scripts that support training techniques such as Tensor Parallelism and Pipeline Parallelism.

## Getting Started

To get started with this project, follow these steps:

1. Clone this project to your local machine: `git clone https://github.com/Barber0/LLM-pretrain.git`
2. Install the required dependencies: `pip install transformers tokenizers datasets deepspeed`
3. Run the pretraining script to train the language model: `deepspeed --num_gpus=1 myllm_train.py`

Please note that to successfully run the pretraining and fine-tuning scripts, you need to prepare appropriate training datasets and set the corresponding paths and parameters in the configuration file. Several DataLoader options are provided for the following datasets:

1. [HuggingFaceH4/self_instruct](https://huggingface.co/datasets/HuggingFaceH4/self_instruct)
2. [WebText](https://huggingface.co/datasets/openwebtext)
3. [Bookcorpus](https://huggingface.co/datasets/bookcorpus)
4. [Wikipedia](https://huggingface.co/datasets/wikipedia)

## License

This project is licensed under the [MIT License](LICENSE). See the license file for more information.

---

We hope that this project helps you achieve better results in language model pretraining and fine-tuning! If you have any questions or need further assistance, please feel free to contact us.

Wishing you success in your natural language processing journey!
