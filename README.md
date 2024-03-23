# Mistral Transformer

This repository contains minimal code to run our 7B model and to finetune it.\
Blog: [https://mistral.ai/news/announcing-mistral-7b/](https://mistral.ai/news/announcing-mistral-7b/)\
Discord: [https://discord.com/invite/mistralai](https://discord.com/invite/mistralai)

## Getting started

### Download the model

```
wget https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

**Note**: The unzipped folder can be used as `initial_model_path:` in the training config.

### Installation Hackathon

Upon running the [Docker container](http://ghcr.io/coreweave/ml-containers/torch-extras:a5a99e8-nccl-cuda12.2.2-ubuntu22.04-nccl2.19.3-1-torch2.2.0-vision0.17.0-audio2.2.0), all necessary dependencies can be installed with:

```bash
pip install -r requirements_hackathon.txt
```

## Using the trained model

### Deployment

The `deploy` folder contains code to build a [vLLM](https://github.com/vllm-project/vllm) image with the required dependencies to serve the Mistral AI model. In the image, the [transformers](https://github.com/huggingface/transformers/) library is used instead of the reference implementation. To build it:

```bash
docker build deploy --build-arg MAX_JOBS=8
```

Instructions to run the image can be found in the [official documentation](https://docs.mistral.ai/quickstart).

### Run the model

```
python -m main demo /path/to/mistral-7B-v0.1/
# To give your own prompts
python -m main interactive /path/to/mistral-7B-v0.1/
```

Change `temperature` or `max_tokens` using:

```
python -m main interactive /path/to/mistral-7B-v0.1/ --max_tokens 256 --temperature 1.0
```

If you want a self-contained implementation, look at `one_file_ref.py`, or run it with

```
python -m one_file_ref /path/to/mistral-7B-v0.1/

This is a test of the emergency broadcast system. This is only a test.

If this were a real emergency, you would be told what to do.

This is a test
=====================
This is another test of the new blogging software. I’m not sure if I’m going to keep it or not. I’m not sure if I’m going to keep
=====================
This is a third test, mistral AI is very good at testing. 🙂

This is a third test, mistral AI is very good at testing. 🙂

This
=====================
```

To run logits equivalence through chunking and sliding window, launch

```
python -m test_generate
```

## Fine-tune the model

### Data

Data must be stored in jsonl format files.

You can build two types of data files:

- _Pretrain_: plain text data stored in the `"text"` key. E.g:

```jsonl
{"text": "Text contained in document n°1"}
{"text": "Text contained in document n°2"}
```

- _Instruct_: conversational data stored in the `"interactions"` key in the form of a list. Each list item is a dictionary containing the `"text"` and `"is_user"` keys. `is_user` is a boolean, if it is equal to True the loss will not be calculated on these tokens. E.g.:

```jsonl
{"interactions": [{"is_user": true, "text": "User interaction n°1 contained in document n°1"}, {"is_user": false, "text": "Bot interaction n°1 contained in document n°1"}, {"is_user": true, "text": "User interaction n°2 contained in document n°1"}, {"is_user": false, "text": "Bot interaction n°2 contained in document n°1"}]}
{"interactions": [{"is_user": true, "text": "User interaction n°1 contained in document n°2"}, {"is_user": false, "text": "Bot interaction n°1 contained in document n°2"}, {"is_user": true, "text": "User interaction n°2 contained in document n°2"}, {"is_user": false, "text": "Bot interaction n°2 contained in document n°2"}, {"is_user": true, "text": "User interaction n°3 contained in document n°2"}, {"is_user": false, "text": "Bot interaction n°3 contained in document n°2"}]}
```

### LoRA Finetuning

To benefit from a memory-efficient and performant finetuning, we recommend to use [LoRA](https://arxiv.org/abs/2106.09685). The idea is to freeze weights and to only learn 1-2% additional weights in the form of low-rank matrix perturbations.

With proper tuning (carefully calibrated learning rate, rank, LoRA dropout, learning the LoRA weights as well as the normalization layers), LoRA finetuning effectively recovers the performance of full finetuning. We support DDP on top of that, meaning that training speed can be increased on multiple GPUs.

After the training, we merge the LoRA weights: hence, the saved checkpoint is exacly in the same format as one would get with full finetuning. To run a LoRA finetuning on a single GPU, use:

```bash
torchrun --nproc-per-node 1 --master_port $RANDOM -m train reference/7B_lora.yaml
```
