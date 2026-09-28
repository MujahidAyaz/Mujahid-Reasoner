# Mujahid-Reasoner

Mujahid-Reasoner is a custom, small-scale language model project focused on language modeling, reasoning, and code-related tasks. The project explores the complete LLM development pipeline, from data preparation and tokenizer training to model training, inference, and future integration with retrieval-augmented generation (RAG), AI agents, and verified self-improvement.

## Project Overview

The project includes a custom decoder-only Transformer model trained on a large text corpus. It is designed as a learning and engineering project to explore modern LLM architecture, training workflows, and AI system development.

### Model Specifications

| Component          | Configuration                  |
| ------------------ | ------------------------------ |
| Architecture       | Decoder-only Transformer       |
| Parameters         | Approximately 12.9 million     |
| Vocabulary size    | 32,000 tokens                  |
| Hidden size        | 256                            |
| Transformer layers | 6                              |
| Attention heads    | 8                              |
| Key-value heads    | 4                              |
| Context length     | 512 tokens                     |
| Training corpus    | Approximately 1 billion tokens |
| Tokenizer          | Custom Byte-Level BPE          |
| Training precision | BF16                           |

The model uses Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), RMSNorm, and SwiGLU.

## Features

* Custom Byte-Level BPE tokenizer
* Data cleaning, filtering, deduplication, and preprocessing
* Tokenization and binary dataset preparation
* Custom Transformer architecture
* Training engine with checkpointing and validation
* Model inference and text generation
* Configurable training and model settings

## Project Architecture

```text
Mujahid-Reasoner/
├── configs/          # Model, tokenizer, and training configurations
├── data/             # Raw, processed, and tokenized datasets
├── tokenizer/        # Tokenizer implementation and vocabulary
├── src/
│   ├── data/         # Data loading and dataset utilities
│   ├── tokenizer/    # Tokenizer training
│   ├── model/        # Transformer architecture
│   ├── training/     # Training engine and optimization
│   ├── evaluation/   # Model evaluation
│   └── inference/    # Text generation
├── scripts/          # Data preparation and utility scripts
├── experiments/      # Checkpoints and experiment outputs
├── tests/            # Tests
├── README.md
├── pyproject.toml
└── LICENSE
```

## Technology Stack

* Python
* PyTorch
* Hugging Face Tokenizers
* NumPy
* PyYAML
* Git and GitHub

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MujahidAyaz/Mujahid-Reasoner.git
cd Mujahid-Reasoner
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -e .
```

Install the appropriate PyTorch build for your hardware if it is not included in the project dependencies.

### 4. Run inference

Use the project's generation script after configuring the model checkpoint and tokenizer paths.

```bash
python scripts/generate.py
```

A trained checkpoint and the matching tokenizer are required for inference.

## Current Status

The project has completed its initial data preparation, tokenizer training, Transformer implementation, and large-scale training pipeline.

The model has been trained on approximately 1 billion tokens. Its current standalone generation quality is limited, and further evaluation and improvement are ongoing.

## Roadmap

* [ ] Improve model evaluation and generation quality
* [ ] Build a Retrieval-Augmented Generation (RAG) pipeline
* [ ] Develop an AI agent with tool calling and execution workflows
* [ ] Integrate knowledge graphs and GraphRAG
* [ ] Implement a verified self-improvement pipeline
* [ ] Build a web-based user interface
* [ ] Add API support and deployment
* [ ] Publish evaluation results and project documentation

## Future Direction

The long-term goal is to develop Mujahid-Reasoner into an integrated AI system combining a custom language model, document retrieval, tool-using agents, knowledge graphs, and a controlled learning pipeline.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
