# LLM from Scratch — Hands-On Curriculum (PyTorch)

[![Video Thumbnail](https://img.youtube.com/vi/p3sij8QzONQ/0.jpg)](https://youtu.be/p3sij8QzONQ?si=yEuD584cBZRNiUYm)

## Introduction

This repository provides a comprehensive, hands-on curriculum for building, training, and aligning Large Language Models (LLMs) from scratch using PyTorch.

The curriculum is divided into nine distinct parts, designed to guide a user from the fundamental mathematics of a single self-attention mechanism to the construction of a full-scale, modern GPT model, culminating in advanced alignment techniques such as RLHF, PPO, and GRPO.

Each part is self-contained with dedicated code, demos, and tests.

---

## Core Architectural and Training Features

### ✅ Core Transformer

* Self-attention
* Multi-Head Attention (MHA)
* LayerNorm
* Feed-forward networks

### ✅ Architectural Upgrades

* **RMSNorm**
* **RoPE (Rotary Positional Embeddings)**
* **SwiGLU Activation**
* **Grouped-Query Attention (GQA)**
* **KV Cache (Sliding Window + Attention Sink)**

### ✅ Sparse Models

* **Mixture-of-Experts (MoE)** with Top-K gating

### ✅ Scalable Training Pipeline

* BPE Tokenization
* Mixed precision (AMP) + Gradient accumulation
* WarmupCosineLR scheduler
* Full checkpoint restore

### ✅ RLHF Alignment

* SFT (Supervised Fine-Tuning)
* Reward Modeling
* PPO and GRPO

---

## Installation

```bash
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```

---

## Repository Structure

| Part   | Description                                       |
| ------ | ------------------------------------------------- |
| part_0 | Foundations & LLM pipeline overview               |
| part_1 | Core Transformer from scratch                     |
| part_2 | Train a Tiny GPT model                            |
| part_3 | Modern GPT upgrades (RoPE, GQA, KV Cache, SwiGLU) |
| part_4 | Scalable training + checkpointing + BPE           |
| part_5 | Mixture‑of‑Experts (MoE)                          |
| part_6 | Supervised Fine‑Tuning (SFT)                      |
| part_7 | Reward Modeling (RM)                              |
| part_8 | PPO‑based RLHF                                    |
| part_9 | GRPO‑based RLHF                                   |

---

## Commands (Examples)

### Part 1 — Transformer Foundations

```bash
cd part_1
python orchestrator.py
python orchestrator.py --visualize
pytest -q
```

### Part 2 — Tiny GPT

```bash
cd part_2
python orchestrator.py
python train.py --data tiny.txt --steps 300 --sample_every 100
python sample.py --ckpt runs/min-gpt/model_best.pt --prompt "Once upon a time"
```

*(All other commands remain the same as in your provided content — nothing removed.)*

---

## License

GPLv3 — open‑source, copyleft.

---

## Contributing

```bash
git checkout -b feature/MyAwesomeFeature
git commit -m "Add MyAwesomeFeature"
git push origin feature/MyAwesomeFeature
```

Pull requests welcome.
