# MIDP-GLLM: Domain-Adaptive Alignment Graph-aware LLM for Malicious Information Diffusion Prediction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![Transformers 4.55.2](https://img.shields.io/badge/Transformers-4.55.2-red?logo=huggingface)](https://huggingface.co/transformers/)
[![peft 0.16.0](https://img.shields.io/badge/PEFT-0.16.0-purple?logo=huggingface)](https://github.com/huggingface/peft)
[![PyTorch Geometric 2.6.1](https://img.shields.io/badge/PyG-2.6.1-orange?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io)
[![FAISS CPU 1.12.0](https://img.shields.io/badge/FAISS--CPU-1.12.0-green?logo=facebook)](https://github.com/facebookresearch/faiss)

Official implementation of the paper:  
**"MIDP-GLLM: Domain-Adaptive Alignment Graph-aware LLM for Malicious Information Diffusion Prediction"**

> We propose a novel paradigm that integrates Graph-aware Large Language Models (LLMs) to unify semantic depth and structural reasoning, addressing the representation challenges caused by extreme topological sparsity in malicious diffusion.

## 🌟 Key Ideas

- **Domain-Adaptive Token Alignment**: Smoothly maps heterogeneous graph representations into the LLM's native token space through a two-stage mechanism:
  - **Semantic-Cognitive Prototype Alignment**: Establishes semantic anchors via clustering-based contrastive learning to bridge the semantic domain gap between general knowledge and specific intent.
  - **Cross-layer Topological Awareness**: Mitigates the over-smoothing problem in deep graph aggregation through cross-layer contrastive learning, preserving structural discriminability.
- **Graph-aware Reasoning Architecture**: Innovatively redesigns the LLM's internal mechanisms to perceive social structures:
  - **Graph-aware Multi-head Attention (MHGA)**: Explicitly injects social network topology into the attention computation to overcome the LLM's "structural blind spot".
  - **Higher-order Dependency Capture**: Modulates attention operators with structural priors to precisely perceive social influence and homophily.
- **Paradigm Shift**: Unlike traditional methods that fragment feature extraction and sequence prediction, our framework achieves deep integration through the unified internal mechanism of the LLM.
- **Superior Generalization & Efficiency**: Demonstrates significant performance gains across various lightweight LLMs (e.g., Qwen3, DeepSeek, MobileLLM) while maintaining low inference latency.

## 📊 Dataset Reconstruction (Misderdect)

To address the **topological sparsity** of malicious spread found in traditional datasets like Douban, we reconstructed the Misderdect dataset:
1. **Core Corpus Establishment**: Integrated verified rumor and fake news topics flagged by official platforms.
2. **Cascade Modeling**: Established diffusion paths by collecting user comment and repost reactions.
3. **Relationship Enhancement**: Extracted multi-dimensional social ties (Following, Fans, Likes) to construct a structurally rich, multi-relational social graph.

### 🔥 Due to dataset size limitations, the restructured version has been uploaded to [Kaggle](https://www.kaggle.com/datasets/yangzhou32/misderdect).

## ⚠️ Important Requirements for Reproduction

### 🔥 Critical Configuration Requirements:

1. **Library Version Consistency**: 
   > **⚠️ CRITICAL**: All code must use `transformers 4.55.2` version. Using other versions will cause the system to fail. This is essential for proper functionality.

2. **Model Path Configuration**:
   > **⚠️ CRITICAL**: You must manually modify the pre-trained LLM weight paths in the model files to match your local paths, otherwise loading will fail. Pay special attention to the `llm_path` parameter in the model files before running.

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
│   ├── Misderdect/
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
git clone https://github.com/yzhouli/IDP-LLM.git
cd IDP-LLM
pip install -r requirements.txt
python run.py
```
