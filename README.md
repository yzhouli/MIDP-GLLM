# IDP-LLM: Graph-enhanced LLM-driven Information Diffusion Prediction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![Transformers 4.55.2](https://img.shields.io/badge/Transformers-4.55.2-red?logo=huggingface)](https://huggingface.co/transformers/)
[![peft 0.16.0](https://img.shields.io/badge/PEFT-0.16.0-purple?logo=huggingface)](https://github.com/huggingface/peft)
[![PyTorch Geometric 2.6.1](https://img.shields.io/badge/PyG-2.6.1-orange?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io)
[![FAISS CPU 1.12.0](https://img.shields.io/badge/FAISS--CPU-1.12.0-green?logo=facebook)](https://github.com/facebookresearch/faiss)

Official implementation of the paper:  
**"IDP-LLM: Graph-enhanced LLM-driven Information Diffusion Prediction"**  
<!-- Accepted at **SIGIR 2025** -->

> We propose a novel framework that unifies large language models (LLMs) with social graph structures to predict information diffusion cascades through interest space alignment.

## 🌟 Key Ideas

- **Interest Space Alignment**: Aligns user and topic embeddings into a socio-psychologically meaningful space via:
  - **Self-driven pattern**: Clustering-based contrastive learning for user interests.
  - **Group-driven pattern**: Structural consistency between users and their engaged topics.
- **Graph-enhanced LLM**: Injects social graph structure into LLM’s attention mechanism without full fine-tuning (uses LoRA).
- **Unified Training Objective**: Casts diffusion prediction as an autoregressive sequence generation task compatible with LLM pretraining.

## ▶️ Quick Start

### 📁 Project Structure
```bash
├── data/
│   ├── .DS_Store
│   ├── christianity/
│   │   ├── cascades.txt
│   │   ├── edges.txt
│   │   ├── idx2u.pickle
│   │   ├── u2idx.pickle
│   ├── douban/
│   │   ├── cascades.txt
│   │   ├── edges.txt
│   │   ├── idx2u.pickle
│   │   ├── u2idx.pickle
│   ├── memetracker/
│   │   ├── cascades.txt
│   │   ├── edges.txt
│   │   ├── idx2u.pickle
│   │   ├── u2idx.pickle
│   ├── pheme/
│   │   ├── cascades.txt
│   │   ├── edges.txt
│   │   ├── idx2u.pickle
│   │   ├── u2idx.pickle
│   ├── weight/
├── helpers/
│   ├── BaseLoader.py
│   ├── BaseRunner.py
│   ├── __pycache__/
│   │   ├── BaseLoader.cpython-310.pyc
│   │   ├── BaseRunner.cpython-310.pyc
├── layers/
│   ├── Commons.py
│   ├── GraphBuilder.py
│   ├── TransformerBlock.py
│   ├── __pycache__/
│   │   ├── Commons.cpython-310.pyc
│   │   ├── GraphBuilder.cpython-310.pyc
│   │   ├── TransformerBlock.cpython-310.pyc
├── log/
├── models/
│   ├── DyHGCN.py
│   ├── Graph_LLM.py
│   ├── Graph_LLM_Deepseek.py
│   ├── Graph_LLM_GPT2.py
│   ├── Graph_LLM_Llama.py
│   ├── Graph_LLM_MobileLLM.py
│   ├── IDP_LLM.py
│   ├── IDP_LLM_LoRA.py
│   ├── LLMNet.py
│   ├── MIM.py
│   ├── PMRCA.py
│   ├── __pycache__/
│   │   ├── DyHGCN.cpython-310.pyc
│   │   ├── Graph_LLM.cpython-310.pyc
│   │   ├── Graph_LLM_Deepseek.cpython-310.pyc
│   │   ├── Graph_LLM_GPT2.cpython-310.pyc
│   │   ├── Graph_LLM_Llama.cpython-310.pyc
│   │   ├── Graph_LLM_MobileLLM.cpython-310.pyc
│   │   ├── IDP_LLM.cpython-310.pyc
│   │   ├── IDP_LLM_LoRA.cpython-310.pyc
│   │   ├── LLMNet.cpython-310.pyc
│   │   ├── MIM.cpython-310.pyc
│   │   ├── PMRCA.cpython-310.pyc
├── README.md
├── requirements.txt
├── run.py
├── saved/
├── temp.py
├── utils/
│   ├── Constants.py
│   ├── Metrics.py
│   ├── Optim.py
│   ├── Utils.py
│   ├── __pycache__/
│   │   ├── Constants.cpython-310.pyc
│   │   ├── Metrics.cpython-310.pyc
│   │   ├── Optim.cpython-310.pyc
│   │   ├── Utils.cpython-310.pyc

### Installation

```bash
git clone https://github.com/yourname/IDP-LLM.git
cd IDP-LLM
pip install -r requirements.txt
python run.py
