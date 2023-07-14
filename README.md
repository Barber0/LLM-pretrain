# 从0开始的LLM: 大语言模型预训练和微调脚本

本项目提供了一套用于预训练和微调大语言模型的脚本，用于快速开始构建自己的自然语言处理应用或研究项目。

## 背景

随着人工智能和自然语言处理领域的发展，大语言模型如GPT-3.5等已经成为许多应用和研究项目的核心。预训练和微调是训练这些大型模型的常见方法，它们可以通过大规模的文本数据来学习语言的统计特性，并在特定任务上进行微调以提高性能。

本项目旨在简化预训练和微调过程，并提供一组易于使用的脚本，使你能够更轻松地使用大语言模型。

## 功能特性

- 提供了预训练脚本，可用于从大规模文本数据中训练语言模型。
- 使用Pytorch基础算子实现了一个类GPT的大语言模型。
- 提供了分布式预训练的脚本，支持Tensor Parallelism、Pipeline Parallelism等训练方式。

## 快速开始

要开始使用本项目，请按照以下步骤操作：

1. 克隆本项目到本地机器：`git clone https://github.com/your-username/your-repo.git`
2. 安装所需的依赖项：`pip install transformers tokenizers datasets deepspeed`
3. 运行预训练脚本以训练语言模型：`deepspeed --num_gpus=1 myllm_train.py`

请注意，为了成功运行预训练和微调脚本，需要准备好合适的训练数据集，并在配置文件中设置相应的路径和参数。目前已提供如下几个数据集的DataLoader：

1. [HuggingFaceH4/self_instruct](https://huggingface.co/datasets/HuggingFaceH4/self_instruct)
2. [WebText](https://huggingface.co/datasets/openwebtext)
3. [Bookcorpus](https://huggingface.co/datasets/bookcorpus)
4. [Wikipedia](https://huggingface.co/datasets/wikipedia)

## 许可证

本项目采用 [MIT 许可证](LICENSE)。请查阅许可证文件以获取更多信息。

---

我们希望这个项目能够帮助你在语言模型预训练和微调方面取得更好的成果！如果你有任何问题或需要进一步的帮助，请随时联系。

祝你在自然语言处理的旅程中一切顺利！
