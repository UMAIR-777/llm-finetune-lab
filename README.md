# LLM Fine-Tune Lab

An experimental workspace and end-to-end pipeline for fine-tuning open-source Large Language Models (such as **Qwen2.5-Instruct**) using parameter-efficient techniques (**LoRA / QLoRA**). 

This repository covers synthetic dataset generation, ChatML formatting, multi-turn tool calling, and model evaluation on consumer/cloud GPUs (e.g., NVIDIA T4 on Google Colab).

## ✨ Key Features
* **Synthetic Data Generation:** Scripts to generate and deduplicate clean, structured JSON/JSONL datasets using ChatML format.
* **Memory-Efficient Training:** QLoRA and LoRA training notebooks optimized for 16GB VRAM GPUs (Tesla T4) using **Unsloth** and **Hugging Face TRL**.
* **Agentic & Tool-Calling Workflows:** Fine-tuning pipelines focused on structured JSON outputs, function calling, and handling ambiguous user queries with clarifying questions.
* **Base vs. Fine-Tuned Evaluation:** Side-by-side benchmarking scripts to compare base model outputs against merged LoRA weights.

## 📂 Repository Structure
```text
llm-finetune-lab/
├── data/               # Sample & processed JSONL datasets (ChatML format)
├── scripts/            # Synthetic data generation & deduplication scripts
├── notebooks/          # Google Colab training & inference notebooks
├── eval/               # Base vs. fine-tuned model evaluation scripts
├── .gitignore          # Ignores model weights, checkpoints, and .env keys
└── README.md           # Project documentation
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com
cd llm-finetune-lab
```

### 2. Install Dependencies
```bash
pip install -q transformers datasets peft trl accelerate bitsandbytes huggingface_hub
```

### 3. Run Training
Open any notebook inside the `notebooks/` folder in **Google Colab** (select **T4 GPU** runtime) and run the cells step-by-step to:
1. Load a quantized base model (e.g., `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen2.5-3B-Instruct`).
2. Format your `.jsonl` dataset into ChatML.
3. Train LoRA adapters and export merged weights or GGUF files to Hugging Face Hub.

## 🛠️ Tech Stack
* **Frameworks:** PyTorch, Hugging Face (`transformers`, `peft`, `trl`, `datasets`), Unsloth
* **Models:** Qwen2.5-Instruct (1.5B / 3B / 7B)
* **Compute:** Google Colab (NVIDIA Tesla T4)

## 📄 License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
