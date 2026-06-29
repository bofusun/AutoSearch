# AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning

Code for "AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning", ACL Findings 2026 paper.

---

# Introduction

we propose a novel Autonomous Search efficiency framework (AutoSearch) with a self-answering mechanism to achieve both accurate and efficient multi-step retrieval in agentic RAG.
On the one hand, we demonstrate that unrestricted search depth in agentic RAG can lead to redundant search steps, incurring substantial computational cost and latency, while overly restricting search depth often leads to underexploration of complex questions, ultimately constraining accuracy. To address this issue, we first investigate how search depth affects accuracy and find a minimal sufficient search depth that defines an accuracy-efficiency trade-off, jointly determined by question complexity and the agent's capability.
On the other hand, building on this finding, the AutoSearch framework evaluates each search step through self-generated intermediate answers, identifying the minimal sufficient search depth via a self-answering mechanism, thus enabling efficient search by rewarding its attainment while penalizing over-searching. In addition, we introduce reward mechanisms to stabilize search behavior and improve answer quality on complex questions.
Through extensive experiments on multiple benchmarks, AutoSearch demonstrates a superior accuracy-efficiency trade-off, alleviating over-searching while preserving search quality, surpassing prior agentic RAG methods.

![Uploading ca43f85c-f5c9-422d-b70e-3fbde9d0a85f.png…]()

---
# Quick Start

1. Setting up repo
```
git clone https://github.com/bofusun/AutoSearch
```
2. Install Dependencies
```
conda create -n AutoSearch python=3.8
conda activate AutoSearch
cd AutoSearch
pip install -r requirements.txt
```
3. Train

```
cd experiments
bash train_autosearch.sh
```
