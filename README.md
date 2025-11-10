

# **LLM from Scratch — Hands-On Curriculum (PyTorch)**


## **Introduction**

This repository provides a comprehensive, hands-on curriculum for building, training, and aligning Large Language Models (LLMs) from scratch using PyTorch. This project is structured as a progressive educational journey, not a monolithic codebase.

The curriculum is divided into nine distinct parts, designed to guide a user from the fundamental mathematics of a single self-attention mechanism (implemented first in NumPy for clarity) 1 to the construction of a full-scale, modern GPT model.1 It culminates in advanced alignment techniques, including Reinforcement Learning from Human Feedback (RLHF) with both Proximal Policy Optimization (PPO) 1 and Group-Relative Policy Optimization (GRPO).1

Each part is self-contained, with its own dedicated code, demonstrations, and tests. This modular structure allows users to build and understand each component in isolation before combining them into a more complex system. This "first principles" approach is designed to demystify the "black box" of modern generative AI by tracing the logic from its mathematical foundations to a fully functional, aligned model.

## **Core Architectural and Training Features**

This curriculum covers the implementation of the following key components of modern LLMs:

* **Core Transformer:** Building the vanilla Transformer block from first principles, including self-attention, multi-head attention (MHA), layer normalization, and feed-forward networks.\[1, 1\]  
* **Architectural Upgrades:** Implementing state-of-the-art features for enhanced performance and efficiency:  
  * **RMSNorm:** A high-performance alternative to LayerNorm.1  
  * **Rotary Positional Embeddings (RoPE):** A high-performance implementation for relative position encoding.1  
  * **SwiGLU:** An efficient and high-performance FFN activation function.1  
  * **Grouped-Query Attention (GQA):** Supported via the n\_kv\_head parameter in the modern attention module.\[1, 1\]  
  * **KV Caching:** An advanced cache implementation for fast inference, featuring both a **Sliding Window** and an **Attention Sink**.1  
* **Sparse Models:**  
  * **Mixture-of-Experts (MoE):** A drop-in replacement for the FFN layer, featuring TopKGate routing and a load-balancing auxiliary loss.\[1, 1\]  
* **Scalable Training Pipeline:**  
  * **BPE Tokenization:** Training and using a Byte Pair Encoding tokenizer from the tokenizers library.1  
  * **Mixed Precision (AMP) & Gradient Accumulation:** A clean wrapper for torch.cuda.amp and gradient accumulation to simulate larger batch sizes.1  
  * **LR Scheduling:** A per-step WarmupCosineLR scheduler, essential for large-scale training.1  
  * **Atomic Checkpointing:** A robust system for saving and resuming the complete training state (model, optimizer, scheduler, and AMP scaler) that also gracefully handles interruptions.\[1, 1\]  
* **Alignment Pipeline (RLHF):**  
  * **Supervised Fine-Tuning (SFT):** The first alignment step, which teaches the model to follow instructions using prompt-loss masking.\[1, 1\]  
  * **Reward Modeling (RM):** Training a separate, bidirectional Transformer Encoder 1 on a preference dataset 1 using the Bradley-Terry loss 1 to create a "judge" model.  
  * **PPO Alignment:** Using the Reward Model to fine-tune the SFT policy with a PPO objective and a learned value head.\[1, 1\]  
  * **GRPO Alignment:** An alternative, policy-only RLHF algorithm that replaces the learned value head with a group-relative baseline.\[1, 1\]

## **Project-Wide Setup and Installation**

The project is built and tested using Python 3.11. The recommended setup utilizes conda for environment management and pip for package installation.

1. **Create and activate a Conda environment:**  
   Bash  
   conda create \-n llm\_from\_scratch python=3.11  
   conda activate llm\_from\_scratch

   (This setup instruction is sourced from the project's foundational README.md 1).  
2. Install all required dependencies:  
   The requirements.txt file 1 lists all necessary Python packages, including torch, datasets, huggingface-hub, and tokenizers.  
   Bash  
   pip install \-r requirements.txt

   This dependency list indicates that while the model architectures are built "from scratch" as nn.Module subclasses, the project leverages the standard PyTorch and Hugging Face ecosystem for data handling, tokenization, and logging.  
3. (Optional) Developer Environment:  
   The repository includes VS Code configuration files (.vscode/launch.json 1 and .vscode/settings.json 1\) for replicating the original development and debugging environment.

## **Repository Structure Overview**

The project is divided into parts, each in its own directory. Each part builds upon the concepts of the previous ones, forming a logical and progressive syllabus. **All commands provided in the walkthrough below are intended to be run from *within* their respective part\_X/ directory.**

| Directory | Focus & Core Concepts |
| :---- | :---- |
| part\_0/ | **Foundations & Mindset:** The high-level 3-stage LLM pipeline (Pretraining $\\rightarrow$ Finetuning $\\rightarrow$ Alignment).1 |
| part\_1/ | **Core Transformer Architecture:** Self-Attention (NumPy & PyTorch), MHA, FFN, LayerNorm, and the full block.1 |
| part\_2/ | **Training a Tiny GPT:** Assembling a full model, byte-level tokenization, dataset batching, and a full training/sampling loop.1 |
| part\_3/ | **Modernizing the Architecture:** Upgrading the vanilla GPT with RoPE, SwiGLU, RMSNorm, GQA, and KV Caching.1 |
| part\_4/ | **Scaling the Training Pipeline:** BPE Tokenizer, AMP, Gradient Accumulation, LR Scheduler, and robust Checkpointing.1 |
| part\_5/ | **Mixture-of-Experts (MoE):** Implementing a sparse FFN layer with Top-K gating and load-balancing loss.1 |
| part\_6/ | **Supervised Fine-Tuning (SFT):** Alignment via instruction-following, using prompt/response formatting and loss masking.1 |
| part\_7/ | **Reward Modeling (RM):** Training a model on preference data to output a scalar "reward" score.1 |
| part\_8/ | **RLHF with PPO:** Using the RM to fine-tune the SFT policy with a PPO objective and a learned value head.1 |
| part\_9/ | **RLHF with GRPO:** An alternative, policy-only RLHF algorithm using a group-relative baseline.1 |

---

## **Curriculum Walkthrough: From Transformer to RLHF**

Below is a detailed breakdown of each part, its key concepts, and the exact commands to run its demos and tests.

### **Part 0: Foundations & Mindset**

This introductory part is conceptual and sets the stage for the project.

* **Conceptual Overview** 1:  
  * **0.1:** Understanding the high-level LLM training pipeline:  
    1. **Pretraining:** Large-scale, self-supervised learning on raw text.  
    2. **Finetuning:** Supervised Fine-Tuning (SFT) on high-quality, curated examples (see Part 6).  
    3. **Alignment:** Aligning the model with human preferences, typically via RLHF (see Parts 7-9).  
  * **0.2:** Hardware and software environment setup (covered in the Project-Wide Setup section above).

### **Part 1: Core Transformer Architecture**

This part builds the fundamental "LEGO brick" of the entire project: the Transformer block, from first principles.

* **Conceptual Overview:**  
  * **1.1 Positional Embeddings:** Implementing both absolute learned and sinusoidal embeddings.1  
  * **1.2 Self-Attention (NumPy):** Implementing the raw math of self-attention using NumPy and "tiny numbers" to trace the matrix operations.1  
  * **1.3 Single Attention Head (PyTorch):** Re-implementing the attention mechanism in PyTorch.1  
  * **1.4 Multi-Head Attention (MHA):** Scaling to MHA, including splitting, concatenation, and projections.1 A demo is provided to trace the intermediate tensor shapes.\[1, 1\]  
  * **1.5 Feed-Forward Network (FFN):** The standard MLP block (using GELU) that follows the attention layer.1  
  * **1.6 Transformer Block:** Assembling all components: LN \-\> MHA \-\> Residual \-\> LN \-\> FFN \-\> Residual.1  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs all demos and tests for Part 1\. |
| attn\_numpy\_demo.py | (1.2) Self-attention math from scratch in NumPy.1 |
| single\_head.py | (1.3) A single self-attention head in PyTorch.1 |
| multi\_head.py | (1.4) The full Multi-Head Self-Attention module.1 |
| ffn.py | (1.5) The Feed-Forward Network (FFN/MLP) block.1 |
| block.py | (1.6) The complete Transformer block.1 |
| pos\_encoding.py | (1.1) Learned and Sinusoidal positional encodings.1 |
| demo\_mha\_shapes.py | A demo that logs all intermediate tensor shapes during an MHA pass.1 |
| demo\_visualize\_multi\_head.py | A demo that saves attention heatmaps to /out.1 |
| tests/test\_attn\_math.py | Validates the PyTorch single\_head module against the NumPy demo.1 |
| tests/test\_causal\_mask.py | Verifies the behavior of the causal mask helper.1 |

This part demonstrates a valuable "Rosetta Stone" validation approach: the test\_attn\_math.py 1 explicitly mirrors and validates the optimized PyTorch single\_head 1 implementation against the simple, human-readable attn\_numpy\_demo.py 1 ground truth.
'''
* Usage 1:  
  All commands must be run from within the part\_1/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_1

  \# Run core demos (NumPy, MHA shapes) and unit tests  
  python orchestrator.py

  \# Run all demos \+ generate visualization images (saves to part\_1/out/)  
  python orchestrator.py \--visualize

  \# Run tests directly  
  pytest \-q
'''
### **Part 2: Training a Tiny GPT**

This part assembles the blocks from Part 1 into a complete GPT model and implements the full training and generation pipeline.

* **Conceptual Overview:**  
  * **2.1 Tokenization:** A simple, byte-level tokenizer (ByteTokenizer) is used, where the vocabulary size is 256 (one for each byte value).1  
  * **2.2 Dataset & Batching:** ByteDataset holds raw bytes from a text file 1 and serves up (x, y) blocks.1 This class correctly implements the crucial next-token-prediction "shift" (where y is x shifted by one token), which is validated by test\_dataset\_shift.py.1  
  * **2.3/2.4 Training Loop:** A full training loop from scratch (no Trainer API) is implemented, including an optimizer and loss calculation.1  
  * **2.5 Sampling:** A generation script (sample.py) 1 is provided, including helpers for temperature, top-k, and top-p (nucleus) sampling.1  
  * **Model:** The GPT class (model\_gpt.py) assembles the tok\_emb, pos\_emb, and a list of Blocks into a full, autoregressive, decoder-only Transformer.1  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs a quick smoke-train, evaluation, and sample demo. |
| tokenizer.py | (2.1) The byte-level tokenizer.1 |
| dataset.py | (2.2) The dataset and batching class with label shifting.1 |
| model\_gpt.py | The complete "tiny GPT" model architecture.1 |
| train.py | (2.3/2.4) The training loop script.1 |
| sample.py | (2.5) The text generation/sampling script.1 |
| utils.py | (2.5) Sampling helpers (top-k, top-p).1 |
| eval\_loss.py | (2.6) Evaluates loss on a validation set.1 |
| tiny.txt, tiny\_hi.txt | Sample training data.\[1, 1\] |

* Usage 1:  
  All commands must be run from within the part\_2/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_2

  \# Run the full smoke-test (train, then sample) via the orchestrator  
  python orchestrator.py

  \# \--- Or run steps manually \---

  \# 1\. Train a model (e.g., on the tiny dataset for 300 steps)  
  python train.py \--data tiny.txt \--steps 300 \--sample\_every 100

  \# 2\. Sample from your trained checkpoint  
  \# (Assuming default out\_dir='runs/min-gpt')  
  python sample.py \--ckpt runs/min-gpt/model\_best.pt \--prompt "Once upon a time"

### **Part 3: Modernizing the Architecture**

This part upgrades the "vanilla" GPT from Part 2 with modern, high-performance components common in state-of-the-art models.

* **Conceptual Overview:**  
  * **3.1 RMSNorm:** Replaces nn.LayerNorm with RMSNorm, which is simpler and often faster.1  
  * **3.2 RoPE:** Replaces learned absolute positional embeddings with Rotary Positional Embeddings, a high-performance relative embedding, implemented via RoPECache.1  
  * **3.3 SwiGLU:** Replaces the GELU-based FFN with a more expressive SwiGLU FFN.1  
  * **3.4/3.6 KV Cache:** Implements an efficient KV Cache for fast autoregressive generation. This is a sophisticated RollingKV implementation 1 that features:  
    * **Rolling Buffer:** Evicts old tokens to keep the cache size fixed.  
    * **Attention Sink:** Always preserves the first sink tokens (e.g., BOS token) to maintain context, a non-trivial optimization.\[1, 1\]  
  * **Grouped-Query Attention (GQA):** The CausalSelfAttentionModern 1 and GPTModern 1 modules accept an n\_kv\_head argument. This allows n\_kv\_head to be set lower than n\_head (e.g., n\_head=32, n\_kv\_head=8), enabling GQA. The code correctly projects Q to n\_head and K/V to n\_kv\_head.1  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs tests and a small generation demo. |
| model\_modern.py | GPTModern wrapper with feature flags for all new components.1 |
| block\_modern.py | The Transformer block updated to use modern components.1 |
| attn\_modern.py | Attention layer with RoPE, GQA, sliding window, sink, and KV cache logic.1 |
| rmsnorm.py | (3.1) RMSNorm implementation.1 |
| rope\_custom.py | (3.2) RoPECache and apply\_rope\_single implementation.1 |
| swiglu.py | (3.3) SwiGLU FFN implementation.1 |
| kv\_cache.py | (3.4) KVCache and RollingKV (sink \+ window) implementation.1 |
| demo\_generate.py | A simple demo showing KV cache and sliding window in action.1 |
| tests/test\_rmsnorm.py | Unit test for rmsnorm.1 |
| tests/test\_rope\_apply.py | Unit test for rope.1 |
| tests/test\_kvcache\_shapes.py | Unit test for kvcache.1 |

* Usage 1:  
  All commands must be run from within the part\_3/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_3

  \# Run tests and the generation demo  
  python orchestrator.py \--demo

  \# \--- Or run steps manually \---

  \# 1\. Run the unit tests  
  pytest \-q

  \# 2\. Run the generation demo with all features enabled  
  python demo\_generate.py \--rmsnorm \--rope \--swiglu \--sliding\_window 64 \--sink 4

### **Part 4: Scaling the Training Pipeline**

This part moves from a toy script to a professional-grade training pipeline, implementing features essential for large-scale, stable training.

* **Conceptual Overview:**  
  * **4.1 BPE Tokenizer:** Implements a Byte Pair Encoding (BPE) tokenizer by wrapping tokenizers 1, which is far more efficient than the Part 2 byte-level tokenizer.  
  * **4.2 AMP & Grad Accum:** Implements a helper class AmpGrad that cleanly wraps torch.cuda.amp.autocast, GradScaler, and gradient accumulation logic.1  
  * **4.3 LR Scheduler:** Implements a WarmupCosineLR scheduler that updates *per step*, not per epoch, which is standard for large-scale pretraining.1  
  * **4.4 Checkpointing:** A robust save\_checkpoint / load\_checkpoint system 1 that atomically saves and loads all necessary states: model weights, optimizer state, scheduler state, and the GradScaler state.  
  * **4.5 Logging:** A flexible logging backend that can be toggled between TensorBoard, WandB, or a NoopLogger.1  
  * **Training Loop:** The main train.py script 1 integrates all these components. It is designed for robustness, even including signal handlers for SIGINT and SIGTERM to save a final checkpoint upon interruption.

This robust checkpointing system \[1, 1\] is critical for long-running jobs, preventing catastrophic failure by saving the complete training state, including optimizer and AMP scaler states 1, and allowing for a seamless resume.

* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs unit tests and a smoke-test of the full training pipeline. |
| train.py | The core training script integrating all scaling components.1 |
| sample.py | Loads a Part 4 checkpoint (model \+ tokenizer) for generation.1 |
| tokenizer\_bpe.py | (4.1) BPE tokenizer training and loading.1 |
| dataset\_bpe.py | (4.1) Streaming dataset for BPE tokens.1 |
| amp\_accum.py | (4.2) AmpGrad helper for mixed precision and gradient accumulation.1 |
| lr\_scheduler.py | (4.3) WarmupCosineLR scheduler.1 |
| checkpointing.py | (4.4) Save/resume functions for the complete training state.1 |
| logger.py | (4.5) Logging backends (TensorBoard, etc.).1 |
| tests/test\_tokenizer\_bpe.py | Unit test for BPE.1 |
| tests/test\_scheduler.py | Unit test for LR scheduler.1 |
| tests/test\_resume\_shapes.py | Unit test for checkpoint resuming.1 |

* Usage 1:  
  All commands must be run from within the part\_4/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_4

  \# Run the full smoke-test (tests, train, sample) via the orchestrator  
  python orchestrator.py \--demo

  \# \--- Or run steps manually \---

  \# 1\. Run the training script (a quick overfit on the tiny file)  
  \# This will train a new BPE tokenizer and save it to runs/part4-demo/tokenizer  
  python train.py \--data../part\_2/tiny.txt \--out runs/part4-demo \--bpe \--vocab\_size 8000 \\  
                  \--steps 300 \--batch\_size 16 \--block\_size 128 \\  
                  \--n\_layer 2 \--n\_head 2 \--n\_embd 128 \\  
                  \--mixed\_precision \--grad\_accum\_steps 2 \--log tensorboard

  \# 3\. Monitor with TensorBoard  
  tensorboard \--logdir=runs/part4-demo

  \# 4\. Sample from the final checkpoint  
  python sample.py \--ckpt runs/part4-demo/model\_last.pt \--prompt "Generate a short story"

### **Part 5: Mixture-of-Experts (MoE)**

This part implements a sparse Mixture-of-Experts (MoE) layer, designed as a drop-in replacement for the dense FFN (MLP) block.1

* **Conceptual Overview:**  
  * **Experts:** An ExpertMLP module (using SwiGLU or GELU) that acts as one of several parallel FFNs.1  
  * **Gate/Router:** A TopKGate module that scores each token across all experts, selects the top\_k (e.g., k=1 or k=2) experts, and assigns softmax-weighted scores for routing.1  
  * **Load Balancing Loss:** The TopKGate also computes an auxiliary loss to encourage tokens to be distributed evenly across all experts, preventing expert collapse.1  
  * **Dispatch/Combine:** The main MoE module 1 takes the token embeddings, uses the TopKGate to get indices and weights, dispatches the tokens to their assigned ExpertMLPs, and combines their weighted outputs.  
  * The implementation in moe.py is "single-GPU friendly" and explicitly loops over experts for pedagogical clarity, making the scatter/gather logic easy to follow.1  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs unit tests and an optional MoE forward pass demo. |
| README.md | (5.1/5.3) Concept notes for the MoE layer.1 |
| gating.py | TopKGate module with load-balancing auxiliary loss.1 |
| experts.py | ExpertMLP module.1 |
| moe.py | The main MoE layer that combines the gate and experts.1 |
| block\_hybrid.py | Example of a hybrid block blending dense FFN and MoE outputs.1 |
| demo\_moe.py | A small demo that runs a forward pass and shows routing.1 |
| tests/test\_gate\_shapes.py | Unit test for gate shapes.1 |
| tests/test\_moe\_forward.py | Unit test for MoE forward pass.1 |
| tests/test\_hybrid\_block.py | Unit test for hybrid block.1 |

* Usage 1:  
  All commands must be run from within the part\_5/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_5

  \# Run tests and the MoE demo  
  python orchestrator.py \--demo

  \# \--- Or run steps manually \---

  \# 1\. Run the unit tests  
  pytest \-q

  \# 2\. Run the MoE demo  
  python demo\_moe.py \--experts 4 \--top\_k 1

### **Part 6: Supervised Fine-Tuning (SFT)**

This part begins the "alignment" phase, adapting the pre-trained model to follow instructions and act as a helpful assistant.

* **Conceptual Overview:**  
  * **6.1 Formatting:** A strict template (formatters.py) is defined to structure data as \<s\> \#\#\# Instruction:... \#\#\# Response:... \</s\>.1 This teaches the model the "turn-taking" format.  
  * **6.2 Loss Masking:** This is the core concept of SFT. The SFTCollator 1 tokenizes the full (prompt \+ response) text, but then sets the labels (y) for all prompt tokens to \-100. This special value is ignored by PyTorch's CrossEntropyLoss. The result is that the model's gradients are *only* calculated based on its ability to predict the Response tokens.  
  * **6.3 Curriculum:** A LengthCurriculum sampler is provided to optionally sort the training data by prompt length (short to long) to improve training stability.1  
  * **6.4 Evaluation:** Simple metrics like Exact Match (EM) and Token F1 are provided for evaluation.1  
  * **Pipeline:** The SFT training script 1 demonstrates a clear dependency chain: it loads the GPTModern model from Part 3 and the pre-trained checkpoint and tokenizer from Part 4\.  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs unit tests and a tiny SFT demo (train \+ sample). |
| train\_sft.py | The main SFT training loop.1 |
| sample\_sft.py | Samples from a trained SFT checkpoint.1 |
| formatters.py | (6.1) Defines the \#\#\# Instruction: / \#\#\# Response: template.1 |
| dataset\_sft.py | Loads an instruction dataset (e.g., from HF) into (prompt, response) pairs.1 |
| collator\_sft.py | (6.2) The SFT collator that applies prompt-loss-masking.1 |
| curriculum.py | (6.3) The length-based curriculum sampler.1 |
| evaluate.py | (6.4) Simple EM and F1 evaluation metrics.1 |
| tests/test\_formatter.py | Unit test for formatting.1 |
| tests/test\_masking.py | Unit test for loss masking.1 |

* Usage 1:  
  All commands must be run from within the part\_6/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_6

  \# Run tests and the tiny SFT demo  
  python orchestrator.py \--demo

  \# \--- Or run steps manually (assumes Part 4 checkpoint exists) \---

  \# 1\. Run the SFT training loop  
  \# This loads the Part 4 model and fine-tunes it  
  python train\_sft.py \--ckpt../part\_4/runs/part4-demo/model\_last.pt \\  
                      \--out runs/sft-demo \--steps 300 \--batch\_size 8 \\  
                      \--n\_layer 2 \--n\_head 2 \--n\_embd 128 \\  
                      \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

  \# 2\. Sample from your new SFT-aligned model  
  python sample\_sft.py \--ckpt runs/sft-demo/model\_last.pt \\  
                       \--n\_layer 2 \--n\_head 2 \--n\_embd 128 \\  
                       \--bpe\_dir../part\_4/runs/part4-demo/tokenizer \\  
                       \--prompt "What are the three primary colors?"

### **Part 7: Reward Modeling (RM)**

This part builds the "judge" model, which learns human preferences. This is the first component of RLHF.

* **Conceptual Overview:**  
  * **7.1 Data:** Loads a preference dataset (e.g., Anthropic HH-RLHF) containing (prompt, chosen, rejected) triplets.1  
  * **7.2 Model:** Implements a RewardModel.1 This model is architecturally distinct from the GPT decoder; it is a **Transformer Encoder** (bidirectional) that ingests a full text sequence (e.g., prompt \+ chosen) and outputs a *single scalar value* representing its "reward" or "goodness" score. This bidirectional approach is appropriate as the model is *scoring* a complete text, not generating it autoregressively.  
  * **7.3 Loss:** Implements the **Bradley-Terry pairwise loss**.1 The model computes r\_pos \= RewardModel(prompt, chosen) and r\_neg \= RewardModel(prompt, rejected). The loss, $ \- \\log(\\sigma(r\_{pos} \- r\_{neg})) $, trains the model to maximize the score gap between the chosen and rejected responses.1  
  * **7.4 Evaluation:** The RM is evaluated by its *accuracy*: the percentage of (chosen, rejected) pairs where it correctly assigns a higher score to the chosen response.1  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs unit tests and a tiny RM training demo. |
| train\_rm.py | The main RM training loop.1 |
| eval\_rm.py | (7.4) Evaluates the RM's accuracy on a preference set.1 |
| data\_prefs.py | (7.1) Loads the preference dataset.1 |
| collator\_rm.py | Tokenizes and collates preference pairs into (pos, neg) batches.1 |
| model\_reward.py | (7.2) The Transformer Encoder-based RewardModel.1 |
| loss\_reward.py | (7.3) bradley\_terry\_loss and margin\_ranking\_loss.1 |
| tests/test\_bt\_loss.py | Unit test for the loss.1 |
| tests/test\_reward\_forward.py | Unit test for model forward pass.1 |

* Usage 1:  
  All commands must be run from within the part\_7/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_7

  \# Run tests and the tiny RM demo  
  python orchestrator.py \--demo

  \# \--- Or run steps manually (assumes Part 4 tokenizer exists) \---

  \# 1\. Run the RM training loop  
  python train\_rm.py \--steps 300 \--batch\_size 8 \--loss bt \\  
                     \--n\_layer 2 \--n\_head 2 \--n\_embd 128 \\  
                     \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

  \# 2\. Evaluate the trained RM's accuracy on the test set  
  python eval\_rm.py \--ckpt runs/rm-demo/model\_last.pt \--split test\[:100\] \\  
                    \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

### **Part 8: RLHF with Proximal Policy Optimization (PPO)**

This part implements the full PPO algorithm, using the Part 7 Reward Model to optimize the Part 6 SFT Policy.

* **Conceptual Overview** 1:  
  * **8.1 Policy:** A PolicyWithValue model is created.1 This model loads the SFT (Part 6\) model weights and adds a new, randomly initialized "value head" (a linear layer) that outputs a scalar value. For pedagogical simplicity, this value head is placed on top of the LM logits (vocab\_size \-\> 1\) rather than the hidden states, to avoid modifying the core GPTModern internals.1  
  * **8.2 Rollout:** In a loop, the policy generates responses to prompts. These "rollouts" are then scored by the RM (Part 7\) to get a reward.\[1, 1\]  
  * **8.3 PPO Objective:** The ppo\_losses function 1 calculates the full PPO objective, which involves:  
    1. **Advantage:** Advantage \= Reward \- Value.  
    2. **Policy Loss:** A "clipped" objective that encourages actions with a positive advantage.  
    3. **Value Loss:** An MSE loss to train the value head to predict the expected reward.  
  * **8.4 Training Loop:** The main script 1 orchestrates the complex on-policy loop: Rollout \-\> Score \-\> Calculate Advantage \-\> Update Policy & Value.1 A frozen *reference policy* (the original SFT model) is used to compute a KL penalty, keeping the policy from diverging too far.1  
* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs unit tests and a tiny PPO demo. |
| train\_ppo.py | (8.4) The main PPO-RLHF training loop.1 |
| eval\_ppo.py | Compares the final policy's reward vs. the reference (SFT) policy.1 |
| policy.py | (8.1) The PolicyWithValue model (SFT LM \+ value head).1 |
| rollout.py | (8.2) Prompt formatting, sampling, and logprob/KL utilities.1 |
| ppo\_loss.py | (8.3) The PPO clipped objective, value loss, and entropy loss.1 |
| tests/test\_policy\_forward.py | Unit test for policy forward pass.1 |
| tests/test\_ppo\_loss.py | Unit test for the PPO loss.1 |

* Usage 1:  
  All commands must be run from within the part\_8/ directory.  
  Bash  
  \# Navigate to the directory  
  cd part\_8

  \# Run tests and the tiny PPO demo  
  python orchestrator.py \--demo

  \# \--- Or run steps manually (assumes Part 4, 6, 7 checkpoints exist) \---

  \# 1\. Run the PPO training loop  
  python train\_ppo.py \--policy\_ckpt../part\_6/runs/sft-demo/model\_last.pt \\  
                      \--reward\_ckpt../part\_7/runs/rm-demo/model\_last.pt \\  
                      \--out runs/ppo-demo \--steps 200 \--batch\_size 4 \--resp\_len 128 \\  
                      \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

  \# 2\. Evaluate the new PPO-aligned policy  
  python eval\_ppo.py \--policy\_ckpt runs/ppo-demo/model\_last.pt \\  
                     \--reward\_ckpt../part\_7/runs/rm-demo/model\_last.pt \\  
                     \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

### **Part 9: RLHF with Group-Relative Policy Optimization (GRPO)**

This final part implements an alternative, simpler RLHF algorithm that avoids the need for a learned value head.

* **Conceptual Overview** 1:  
  * **9.1 Group-Relative Baseline:** This method (train\_grpo.py 1) generates k (e.g., 4\) completions for *each* prompt. It computes the *mean reward* of this group.  
  * **9.2 Advantage Calculation:** The advantage for each completion is its (reward \- group mean reward). This group-relative baseline replaces the learned value network from Part 8, avoiding the instability of training a separate value head.1  
  * **9.3 Policy-Only Objective:** The loss function (ppo\_policy\_only\_losses 1) is a PPO-style clipped objective that *only* updates the policy. It does *not* have a value loss component.  
  * **9.4 KL Regularization:** A simple, explicit KL penalty ($KL(\\pi\_{\\text{policy}} | \\pi\_{\\text{ref}})$) is added directly to the loss 1 to keep the policy from diverging from the SFT model.

This section highlights the curriculum's completeness by not just teaching one method (PPO), but by providing a modern, comparative alternative (GRPO), allowing users to explore different alignment techniques.

* **Key Files** 1:

| File | Purpose |
| :---- | :---- |
| orchestrator.py | Runs unit tests and a tiny GRPO demo. |
| train\_grpo.py | (9.1) The main GRPO-RLHF training loop.1 |
| grpo\_loss.py | (9.3) The policy-only clipped objective \+ explicit KL penalty.1 |
| eval\_ppo.py | (Shared) Evaluates the final policy's reward.1 |
| policy.py | (Shared) The PolicyWithValue model (value head is *ignored*).1 |
| rollout.py | (Shared) Sampling and logprob utilities.1 |
| tests/test\_grpo\_loss.py | Unit test for the GRPO loss.1 |

* Usage 1:  
  All commands must be run from within the part\_9/ directory. (Note: The orchestrator script 1 references running from part\_8/, but the file structure implies part\_9/. The commands below are corrected for the Part 9 context).  
  Bash  
  \# Navigate to the directory  
  cd part\_9

  \# Run tests and the tiny GRPO demo  
  python orchestrator.py \--demo

  \# \--- Or run steps manually (assumes Part 4, 6, 7 checkpoints exist) \---

  \# 1\. Run the GRPO training loop  
  python train\_grpo.py \--group\_size 4 \\  
                       \--policy\_ckpt../part\_6/runs/sft-demo/model\_last.pt \\  
                       \--reward\_ckpt../part\_7/runs/rm-demo/model\_last.pt \\  
                       \--out runs/grpo-demo \--steps 200 \--batch\_prompts 4 \--resp\_len 128 \\  
                       \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

  \# 2\. Evaluate the new GRPO-aligned policy  
  python eval\_ppo.py \--policy\_ckpt runs/grpo-demo/model\_last.pt \\  
                     \--reward\_ckpt../part\_7/runs/rm-demo/model\_last.pt \\  
                     \--bpe\_dir../part\_4/runs/part4-demo/tokenizer

## **License**

This project is licensed under the **GNU General Public License v3 (GPLv3)**, as detailed in the LICENSE file.1

The GPLv3 is a "copyleft" license, which means that any derivative works or distributed software that incorporates this code must also be released under the GPLv3 license, ensuring the code remains free and open-source. As the license preamble states, "you must pass on to the recipients the same freedoms that you received... You must make sure that they, too, receive or can get the source code".1

## **Contributing**

Contributions are welcome. This project follows a standard open-source workflow. Please feel free to:

1. **Fork** the repository.  
2. Create a new **feature branch** (git checkout \-b feature/MyAwesomeFeature).  
3. **Commit** your changes (git commit \-m 'Add MyAwesomeFeature').  
4. **Push** to your branch (git push origin feature/MyAwesomeFeature).  
5. Open a **Pull Request**.

#### **Works cited**

1. vivekkalyanarangan30/llm\_from\_scratch
