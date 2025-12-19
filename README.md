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

## 📈 Main Results (Information Diffusion Prediction)
We compare IDP-LLM against seven baselines: [DyHGCN](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_21), [MS-HGAT](https://ojs.aaai.org/index.php/AAAI/article/view/20334), [MIM](https://ieeexplore.ieee.org/abstract/document/10994219/), [DSHCL](https://ieeexplore.ieee.org/abstract/document/11062122/), [SILN](https://dl.acm.org/doi/abs/10.1145/3711896.3736925), [Ghidorah](https://ojs.aaai.org/index.php/AAAI/article/view/33470), and [PMRCA](https://dl.acm.org/doi/abs/10.1145/3726302.3729883). We also evaluate with lightweight LLMs: [GPT-2](https://huggingface.co/openai-community/gpt2-large), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), and [MobileLLM-R1.5-950M](https://huggingface.co/facebook/MobileLLM-R1.5-950M).
### - Information Diffusion Prediction
![Loading... Please check results/model_comparison.png.](results/model_comparison.png)
### - Generalization Evaluation of Different LLMs
![Loading... Please check results/model_LLM_eval.png.](results/model_LLM_eval.png)
## ▶️ Quick Start

### 📁 Project Structure
```bash
├── data/    ## Public Datasets
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
│   ├── BaseLoader.py   # Dataset Loading and Processing
│   ├── BaseRunner.py   # Model Training, Validation, and Testing
├── layers/
│   ├── Commons.py
│   ├── GraphBuilder.py
│   ├── TransformerBlock.py
├── log/
├── models/
│   ├── DyHGCN.py  # DyHGCN Model
│   ├── Graph_LLM.py
│   ├── Graph_LLM_Deepseek.py
│   ├── Graph_LLM_GPT2.py
│   ├── Graph_LLM_Llama.py
│   ├── Graph_LLM_MobileLLM.py
│   ├── IDP_LLM.py
│   ├── IDP_LLM_LoRA.py  # IDP-LLM Model, Before use, **please verify that the llm_path parameter is correct.**
│   ├── LLMNet.py
│   ├── MIM.py     # MIM Model
│   ├── PMRCA.py   # PMRCA Model
├── utils/
│   ├── Constants.py
│   ├── Metrics.py
│   ├── Optim.py
│   ├── Utils.py
├── README.md
├── requirements.txt
├── run.py
├── saved/
```

### Installation

```bash
git clone https://github.com/yourname/IDP-LLM.git
cd IDP-LLM
pip install -r requirements.txt
python run.py
```
