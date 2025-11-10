LLM from Scratch — Hands-On Curriculum (PyTorch)
(https://img.youtube.com/vi/p3sij8QzONQ/0.jpg)](https://youtu.be/p3sij8QzONQ?si=yEuD584cBZRNiUYm)

Introduction
This repository provides a comprehensive, hands-on curriculum for building, training, and aligning Large Language Models (LLMs) from scratch using PyTorch. This project is structured as a progressive educational journey, not a monolithic codebase.

The curriculum is divided into nine distinct parts, designed to guide a user from the fundamental mathematics of a single self-attention mechanism (implemented first in NumPy for clarity)  to the construction of a full-scale, modern GPT model. It culminates in advanced alignment techniques, including Reinforcement Learning from Human Feedback (RLHF) with both Proximal Policy Optimization (PPO)  and Group-Relative Policy Optimization (GRPO).   

Each part is self-contained, with its own dedicated code, demonstrations, and tests. This modular structure allows users to build and understand each component in isolation before combining them into a more complex system. This "first principles" approach is designed to demystify the "black box" of modern generative AI by tracing the logic from its mathematical foundations to a fully functional, aligned model.

Core Architectural and Training Features
This curriculum covers the implementation of the following key components of modern LLMs:

Core Transformer: Building the vanilla Transformer block from first principles, including self-attention, multi-head attention (MHA), layer normalization, and feed-forward networks.   

Architectural Upgrades: Implementing state-of-the-art features for enhanced performance and efficiency:

RMSNorm: A high-performance alternative to LayerNorm.   

Rotary Positional Embeddings (RoPE): A high-performance implementation for relative position encoding.   

SwiGLU: An efficient and high-performance FFN activation function.   

Grouped-Query Attention (GQA): Supported via the n_kv_head parameter in the modern attention module.   

KV Caching: An advanced cache implementation for fast inference, featuring both a Sliding Window and an Attention Sink.   

Sparse Models:

Mixture-of-Experts (MoE): A drop-in replacement for the FFN layer, featuring TopKGate routing and a load-balancing auxiliary loss.   

Scalable Training Pipeline:

BPE Tokenization: Training and using a Byte Pair Encoding tokenizer from the tokenizers library.   

Mixed Precision (AMP) & Gradient Accumulation: A clean wrapper for torch.cuda.amp and gradient accumulation to simulate larger batch sizes.   

LR Scheduling: A per-step WarmupCosineLR scheduler, essential for large-scale training.   

Atomic Checkpointing: A robust system for saving and resuming the complete training state (model, optimizer, scheduler, and AMP scaler) that also gracefully handles interruptions.   

Alignment Pipeline (RLHF):

Supervised Fine-Tuning (SFT): The first alignment step, which teaches the model to follow instructions using prompt-loss masking.   

Reward Modeling (RM): Training a separate, bidirectional Transformer Encoder  on a preference dataset  using the Bradley-Terry loss  to create a "judge" model.   

PPO Alignment: Using the Reward Model to fine-tune the SFT policy with a PPO objective and a learned value head.   

GRPO Alignment: An alternative, policy-only RLHF algorithm that replaces the learned value head with a group-relative baseline.   

Project-Wide Setup and Installation
The project is built and tested using Python 3.11. The recommended setup utilizes conda for environment management and pip for package installation.

Create and activate a Conda environment:

Bash
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
(This setup instruction is sourced from the project's foundational README.md ).   

Install all required dependencies: The requirements.txt file  lists all necessary Python packages, including torch, datasets, huggingface-hub, and tokenizers.   

Bash
pip install -r requirements.txt
This dependency list indicates that while the model architectures are built "from scratch" as nn.Module subclasses, the project leverages the standard PyTorch and Hugging Face ecosystem for data handling, tokenization, and logging.

(Optional) Developer Environment: The repository includes VS Code configuration files (.vscode/launch.json  and .vscode/settings.json ) for replicating the original development and debugging environment.   

Repository Structure Overview
The project is divided into parts, each in its own directory. Each part builds upon the concepts of the previous ones, forming a logical and progressive syllabus. All commands provided in the walkthrough below are intended to be run from within their respective part_X/ directory.

Directory	Focus & Core Concepts
part_0/	
Foundations & Mindset: The high-level 3-stage LLM pipeline (Pretraining → Finetuning → Alignment).

part_1/	
Core Transformer Architecture: Self-Attention (NumPy & PyTorch), MHA, FFN, LayerNorm, and the full block.

part_2/	
Training a Tiny GPT: Assembling a full model, byte-level tokenization, dataset batching, and a full training/sampling loop.

part_3/	
Modernizing the Architecture: Upgrading the vanilla GPT with RoPE, SwiGLU, RMSNorm, GQA, and KV Caching.

part_4/	
Scaling the Training Pipeline: BPE Tokenizer, AMP, Gradient Accumulation, LR Scheduler, and robust Checkpointing.

part_5/	
Mixture-of-Experts (MoE): Implementing a sparse FFN layer with Top-K gating and load-balancing loss.

part_6/	
Supervised Fine-Tuning (SFT): Alignment via instruction-following, using prompt/response formatting and loss masking.

part_7/	
Reward Modeling (RM): Training a model on preference data to output a scalar "reward" score.

part_8/	
RLHF with PPO: Using the RM to fine-tune the SFT policy with a PPO objective and a learned value head.

part_9/	
RLHF with GRPO: An alternative, policy-only RLHF algorithm using a group-relative baseline.

  
Curriculum Walkthrough: From Transformer to RLHF
Below is a detailed breakdown of each part, its key concepts, and the exact commands to run its demos and tests.

Part 0: Foundations & Mindset
This introductory part is conceptual and sets the stage for the project.

Conceptual Overview :   

0.1: Understanding the high-level LLM training pipeline:

Pretraining: Large-scale, self-supervised learning on raw text.

Finetuning: Supervised Fine-Tuning (SFT) on high-quality, curated examples (see Part 6).

Alignment: Aligning the model with human preferences, typically via RLHF (see Parts 7-9).

0.2: Hardware and software environment setup (covered in the Project-Wide Setup section above).

Part 1: Core Transformer Architecture
This part builds the fundamental "LEGO brick" of the entire project: the Transformer block, from first principles.

Conceptual Overview:

1.1 Positional Embeddings: Implementing both absolute learned and sinusoidal embeddings.   

1.2 Self-Attention (NumPy): Implementing the raw math of self-attention using NumPy and "tiny numbers" to trace the matrix operations.   

1.3 Single Attention Head (PyTorch): Re-implementing the attention mechanism in PyTorch.   

1.4 Multi-Head Attention (MHA): Scaling to MHA, including splitting, concatenation, and projections. A demo is provided to trace the intermediate tensor shapes.   

1.5 Feed-Forward Network (FFN): The standard MLP block (using GELU) that follows the attention layer.   

1.6 Transformer Block: Assembling all components: LN -> MHA -> Residual -> LN -> FFN -> Residual.   

Key Files :   

File	Purpose
orchestrator.py	Runs all demos and tests for Part 1.
attn_numpy_demo.py	
(1.2) Self-attention math from scratch in NumPy.

single_head.py	
(1.3) A single self-attention head in PyTorch.

multi_head.py	
(1.4) The full Multi-Head Self-Attention module.

ffn.py	
(1.5) The Feed-Forward Network (FFN/MLP) block.

block.py	
(1.6) The complete Transformer block.

pos_encoding.py	
(1.1) Learned and Sinusoidal positional encodings.

demo_mha_shapes.py	
A demo that logs all intermediate tensor shapes during an MHA pass.

demo_visualize_multi_head.py	
A demo that saves attention heatmaps to /out.

tests/test_attn_math.py	
Validates the PyTorch single_head module against the NumPy demo.

tests/test_causal_mask.py	
Verifies the behavior of the causal mask helper.

  
This part demonstrates a valuable "Rosetta Stone" validation approach: the test_attn_math.py  explicitly mirrors and validates the optimized PyTorch single_head  implementation against the simple, human-readable attn_numpy_demo.py  ground truth.   

Usage : All commands must be run from within the part_1/ directory.   

Bash
# Navigate to the directory
cd part_1

# Run core demos (NumPy, MHA shapes) and unit tests
python orchestrator.py

# Run all demos + generate visualization images (saves to part_1/out/)
python orchestrator.py --visualize

# Run tests directly
pytest -q
Part 2: Training a Tiny GPT
This part assembles the blocks from Part 1 into a complete GPT model and implements the full training and generation pipeline.

Conceptual Overview:

2.1 Tokenization: A simple, byte-level tokenizer (ByteTokenizer) is used, where the vocabulary size is 256 (one for each byte value).   

2.2 Dataset & Batching: ByteDataset holds raw bytes from a text file  and serves up (x, y) blocks. This class correctly implements the crucial next-token-prediction "shift" (where y is x shifted by one token), which is validated by test_dataset_shift.py.   

2.3/2.4 Training Loop: A full training loop from scratch (no Trainer API) is implemented, including an optimizer and loss calculation.   

2.5 Sampling: A generation script (sample.py)  is provided, including helpers for temperature, top-k, and top-p (nucleus) sampling.   

Model: The GPT class (model_gpt.py) assembles the tok_emb, pos_emb, and a list of Blocks into a full, autoregressive, decoder-only Transformer.   

Key Files :   

File	Purpose
orchestrator.py	Runs a quick smoke-train, evaluation, and sample demo.
tokenizer.py	
(2.1) The byte-level tokenizer.

dataset.py	
(2.2) The dataset and batching class with label shifting.

model_gpt.py	
The complete "tiny GPT" model architecture.

train.py	
(2.3/2.4) The training loop script.

sample.py	
(2.5) The text generation/sampling script.

utils.py	
(2.5) Sampling helpers (top-k, top-p).

eval_loss.py	
(2.6) Evaluates loss on a validation set.

tiny.txt, tiny_hi.txt	
Sample training data.

  
Usage : All commands must be run from within the part_2/ directory.   

Bash
# Navigate to the directory
cd part_2

# Run the full smoke-test (train, then sample) via the orchestrator
python orchestrator.py

# --- Or run steps manually ---

# 1. Train a model (e.g., on the tiny dataset for 300 steps)
python train.py --data tiny.txt --steps 300 --sample_every 100

# 2. Sample from your trained checkpoint
# (Assuming default out_dir='runs/min-gpt')
python sample.py --ckpt runs/min-gpt/model_best.pt --prompt "Once upon a time"
Part 3: Modernizing the Architecture
This part upgrades the "vanilla" GPT from Part 2 with modern, high-performance components common in state-of-the-art models.

Conceptual Overview:

3.1 RMSNorm: Replaces nn.LayerNorm with RMSNorm, which is simpler and often faster.   

3.2 RoPE: Replaces learned absolute positional embeddings with Rotary Positional Embeddings, a high-performance relative embedding, implemented via RoPECache.   

3.3 SwiGLU: Replaces the GELU-based FFN with a more expressive SwiGLU FFN.   

3.4/3.6 KV Cache: Implements an efficient KV Cache for fast autoregressive generation. This is a sophisticated RollingKV implementation  that features:   

Rolling Buffer: Evicts old tokens to keep the cache size fixed.

Attention Sink: Always preserves the first sink tokens (e.g., BOS token) to maintain context, a non-trivial optimization.   

Grouped-Query Attention (GQA): The CausalSelfAttentionModern  and GPTModern  modules accept an n_kv_head argument. This allows n_kv_head to be set lower than n_head (e.g., n_head=32, n_kv_head=8), enabling GQA. The code correctly projects Q to n_head and K/V to n_kv_head.   

Key Files :   

File	Purpose
orchestrator.py	Runs tests and a small generation demo.
model_modern.py	
GPTModern wrapper with feature flags for all new components.

block_modern.py	
The Transformer block updated to use modern components.

attn_modern.py	
Attention layer with RoPE, GQA, sliding window, sink, and KV cache logic.

rmsnorm.py	
(3.1) RMSNorm implementation.

rope_custom.py	
(3.2) RoPECache and apply_rope_single implementation.

swiglu.py	
(3.3) SwiGLU FFN implementation.

kv_cache.py	
(3.4) KVCache and RollingKV (sink + window) implementation.

demo_generate.py	
A simple demo showing KV cache and sliding window in action.

tests/test_rmsnorm.py	
Unit test for rmsnorm.

tests/test_rope_apply.py	
Unit test for rope.

tests/test_kvcache_shapes.py	
Unit test for kvcache.

  
Usage : All commands must be run from within the part_3/ directory.   

Bash
# Navigate to the directory
cd part_3

# Run tests and the generation demo
python orchestrator.py --demo

# --- Or run steps manually ---

# 1. Run the unit tests
pytest -q

# 2. Run the generation demo with all features enabled
python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4
Part 4: Scaling the Training Pipeline
This part moves from a toy script to a professional-grade training pipeline, implementing features essential for large-scale, stable training.

Conceptual Overview:

4.1 BPE Tokenizer: Implements a Byte Pair Encoding (BPE) tokenizer by wrapping tokenizers , which is far more efficient than the Part 2 byte-level tokenizer.   

4.2 AMP & Grad Accum: Implements a helper class AmpGrad that cleanly wraps torch.cuda.amp.autocast, GradScaler, and gradient accumulation logic.   

4.3 LR Scheduler: Implements a WarmupCosineLR scheduler that updates per step, not per epoch, which is standard for large-scale pretraining.   

4.4 Checkpointing: A robust save_checkpoint / load_checkpoint system  that atomically saves and loads all necessary states: model weights, optimizer state, scheduler state, and the GradScaler state.   

4.5 Logging: A flexible logging backend that can be toggled between TensorBoard, WandB, or a NoopLogger.   

Training Loop: The main train.py script  integrates all these components. It is designed for robustness, even including signal handlers for SIGINT and SIGTERM to save a final checkpoint upon interruption.   

This robust checkpointing system  is critical for long-running jobs, preventing catastrophic failure by saving the complete training state, including optimizer and AMP scaler states , and allowing for a seamless resume.   

Key Files :   

File	Purpose
orchestrator.py	Runs unit tests and a smoke-test of the full training pipeline.
train.py	
The core training script integrating all scaling components.

sample.py	
Loads a Part 4 checkpoint (model + tokenizer) for generation.

tokenizer_bpe.py	
(4.1) BPE tokenizer training and loading.

dataset_bpe.py	
(4.1) Streaming dataset for BPE tokens.

amp_accum.py	
(4.2) AmpGrad helper for mixed precision and gradient accumulation.

lr_scheduler.py	
(4.3) WarmupCosineLR scheduler.

checkpointing.py	
(4.4) Save/resume functions for the complete training state.

logger.py	
(4.5) Logging backends (TensorBoard, etc.).

tests/test_tokenizer_bpe.py	
Unit test for BPE.

tests/test_scheduler.py	
Unit test for LR scheduler.

tests/test_resume_shapes.py	
Unit test for checkpoint resuming.

  
Usage : All commands must be run from within the part_4/ directory.   

Bash
# Navigate to the directory
cd part_4

# Run the full smoke-test (tests, train, sample) via the orchestrator
python orchestrator.py --demo

# --- Or run steps manually ---

# 1. Run the training script (a quick overfit on the tiny file)
# This will train a new BPE tokenizer and save it to runs/part4-demo/tokenizer
python train.py --data../part_2/tiny.txt --out runs/part4-demo --bpe --vocab_size 8000                 --steps 300 --batch_size 16 --block_size 128                 --n_layer 2 --n_head 2 --n_embd 128                 --mixed_precision --grad_accum_steps 2 --log tensorboard

# 3. Monitor with TensorBoard
tensorboard --logdir=runs/part4-demo

# 4. Sample from the final checkpoint
python sample.py --ckpt runs/part4-demo/model_last.pt --prompt "Generate a short story"
Part 5: Mixture-of-Experts (MoE)
This part implements a sparse Mixture-of-Experts (MoE) layer, designed as a drop-in replacement for the dense FFN (MLP) block.   

Conceptual Overview:

Experts: An ExpertMLP module (using SwiGLU or GELU) that acts as one of several parallel FFNs.   

Gate/Router: A TopKGate module that scores each token across all experts, selects the top_k (e.g., k=1 or k=2) experts, and assigns softmax-weighted scores for routing.   

Load Balancing Loss: The TopKGate also computes an auxiliary loss to encourage tokens to be distributed evenly across all experts, preventing expert collapse.   

Dispatch/Combine: The main MoE module  takes the token embeddings, uses the TopKGate to get indices and weights, dispatches the tokens to their assigned ExpertMLPs, and combines their weighted outputs.   

The implementation in moe.py is "single-GPU friendly" and explicitly loops over experts for pedagogical clarity, making the scatter/gather logic easy to follow.   

Key Files :   

File	Purpose
orchestrator.py	Runs unit tests and an optional MoE forward pass demo.
README.md	
(5.1/5.3) Concept notes for the MoE layer.

gating.py	
TopKGate module with load-balancing auxiliary loss.

experts.py	
ExpertMLP module.

moe.py	
The main MoE layer that combines the gate and experts.

block_hybrid.py	
Example of a hybrid block blending dense FFN and MoE outputs.

demo_moe.py	
A small demo that runs a forward pass and shows routing.

tests/test_gate_shapes.py	
Unit test for gate shapes.

tests/test_moe_forward.py	
Unit test for MoE forward pass.

tests/test_hybrid_block.py	
Unit test for hybrid block.

  
Usage : All commands must be run from within the part_5/ directory.   

Bash
# Navigate to the directory
cd part_5

# Run tests and the MoE demo
python orchestrator.py --demo

# --- Or run steps manually ---

# 1. Run the unit tests
pytest -q

# 2. Run the MoE demo
python demo_moe.py --experts 4 --top_k 1
Part 6: Supervised Fine-Tuning (SFT)
This part begins the "alignment" phase, adapting the pre-trained model to follow instructions and act as a helpful assistant.

Conceptual Overview:

6.1 Formatting: A strict template (formatters.py) is defined to structure data as <s> ### Instruction:... ### Response:... </s>. This teaches the model the "turn-taking" format.   

6.2 Loss Masking: This is the core concept of SFT. The SFTCollator  tokenizes the full (prompt + response) text, but then sets the labels (y) for all prompt tokens to -100. This special value is ignored by PyTorch's CrossEntropyLoss. The result is that the model's gradients are only calculated based on its ability to predict the Response tokens.   

6.3 Curriculum: A LengthCurriculum sampler is provided to optionally sort the training data by prompt length (short to long) to improve training stability.   

6.4 Evaluation: Simple metrics like Exact Match (EM) and Token F1 are provided for evaluation.   

Pipeline: The SFT training script  demonstrates a clear dependency chain: it loads the GPTModern model from Part 3 and the pre-trained checkpoint and tokenizer from Part 4.   

Key Files :   

File	Purpose
orchestrator.py	Runs unit tests and a tiny SFT demo (train + sample).
train_sft.py	
The main SFT training loop.

sample_sft.py	
Samples from a trained SFT checkpoint.

formatters.py	
(6.1) Defines the ### Instruction: / ### Response: template.

dataset_sft.py	
Loads an instruction dataset (e.g., from HF) into (prompt, response) pairs.

collator_sft.py	
(6.2) The SFT collator that applies prompt-loss-masking.

curriculum.py	
(6.3) The length-based curriculum sampler.

evaluate.py	
(6.4) Simple EM and F1 evaluation metrics.

tests/test_formatter.py	
Unit test for formatting.

tests/test_masking.py	
Unit test for loss masking.

  
Usage : All commands must be run from within the part_6/ directory.   

Bash
# Navigate to the directory
cd part_6

# Run tests and the tiny SFT demo
python orchestrator.py --demo

# --- Or run steps manually (assumes Part 4 checkpoint exists) ---

# 1. Run the SFT training loop
# This loads the Part 4 model and fine-tunes it
python train_sft.py --ckpt../part_4/runs/part4-demo/model_last.pt                     --out runs/sft-demo --steps 300 --batch_size 8                     --n_layer 2 --n_head 2 --n_embd 128                     --bpe_dir../part_4/runs/part4-demo/tokenizer

# 2. Sample from your new SFT-aligned model
python sample_sft.py --ckpt runs/sft-demo/model_last.pt                      --n_layer 2 --n_head 2 --n_embd 128                      --bpe_dir../part_4/runs/part4-demo/tokenizer                      --prompt "What are the three primary colors?"
Part 7: Reward Modeling (RM)
This part builds the "judge" model, which learns human preferences. This is the first component of RLHF.

Conceptual Overview:

7.1 Data: Loads a preference dataset (e.g., Anthropic HH-RLHF) containing (prompt, chosen, rejected) triplets.   

7.2 Model: Implements a RewardModel. This model is architecturally distinct from the GPT decoder; it is a Transformer Encoder (bidirectional) that ingests a full text sequence (e.g., prompt + chosen) and outputs a single scalar value representing its "reward" or "goodness" score. This bidirectional approach is appropriate as the model is scoring a complete text, not generating it autoregressively.   

7.3 Loss: Implements the Bradley-Terry pairwise loss. The model computes r_pos = RewardModel(prompt, chosen) and r_neg = RewardModel(prompt, rejected). The loss, $ - \log(\sigma(r_{pos} - r_{neg})) $, trains the model to maximize the score gap between the chosen and rejected responses.   

7.4 Evaluation: The RM is evaluated by its accuracy: the percentage of (chosen, rejected) pairs where it correctly assigns a higher score to the chosen response.   

Key Files :   

File	Purpose
orchestrator.py	Runs unit tests and a tiny RM training demo.
train_rm.py	
The main RM training loop.

eval_rm.py	
(7.4) Evaluates the RM's accuracy on a preference set.

data_prefs.py	
(7.1) Loads the preference dataset.

collator_rm.py	
Tokenizes and collates preference pairs into (pos, neg) batches.

model_reward.py	
(7.2) The Transformer Encoder-based RewardModel.

loss_reward.py	
(7.3) bradley_terry_loss and margin_ranking_loss.

tests/test_bt_loss.py	
Unit test for the loss.

tests/test_reward_forward.py	
Unit test for model forward pass.

  
Usage : All commands must be run from within the part_7/ directory.   

Bash
# Navigate to the directory
cd part_7

# Run tests and the tiny RM demo
python orchestrator.py --demo

# --- Or run steps manually (assumes Part 4 tokenizer exists) ---

# 1. Run the RM training loop
python train_rm.py --steps 300 --batch_size 8 --loss bt                    --n_layer 2 --n_head 2 --n_embd 128                    --bpe_dir../part_4/runs/part4-demo/tokenizer

# 2. Evaluate the trained RM's accuracy on the test set
python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split test[:100]                   --bpe_dir../part_4/runs/part4-demo/tokenizer
Part 8: RLHF with Proximal Policy Optimization (PPO)
This part implements the full PPO algorithm, using the Part 7 Reward Model to optimize the Part 6 SFT Policy.

Conceptual Overview :   

8.1 Policy: A PolicyWithValue model is created. This model loads the SFT (Part 6) model weights and adds a new, randomly initialized "value head" (a linear layer) that outputs a scalar value. For pedagogical simplicity, this value head is placed on top of the LM logits (vocab_size -> 1) rather than the hidden states, to avoid modifying the core GPTModern internals.   

8.2 Rollout: In a loop, the policy generates responses to prompts. These "rollouts" are then scored by the RM (Part 7) to get a reward.   

8.3 PPO Objective: The ppo_losses function  calculates the full PPO objective, which involves:   

Advantage: Advantage = Reward - Value.

Policy Loss: A "clipped" objective that encourages actions with a positive advantage.

Value Loss: An MSE loss to train the value head to predict the expected reward.

8.4 Training Loop: The main script  orchestrates the complex on-policy loop: Rollout -> Score -> Calculate Advantage -> Update Policy & Value. A frozen reference policy (the original SFT model) is used to compute a KL penalty, keeping the policy from diverging too far.   

Key Files :   

File	Purpose
orchestrator.py	Runs unit tests and a tiny PPO demo.
train_ppo.py	
(8.4) The main PPO-RLHF training loop.

eval_ppo.py	
Compares the final policy's reward vs. the reference (SFT) policy.

policy.py	
(8.1) The PolicyWithValue model (SFT LM + value head).

rollout.py	
(8.2) Prompt formatting, sampling, and logprob/KL utilities.

ppo_loss.py	
(8.3) The PPO clipped objective, value loss, and entropy loss.

tests/test_policy_forward.py	
Unit test for policy forward pass.

tests/test_ppo_loss.py	
Unit test for the PPO loss.

  
Usage : All commands must be run from within the part_8/ directory.   

Bash
# Navigate to the directory
cd part_8

# Run tests and the tiny PPO demo
python orchestrator.py --demo

# --- Or run steps manually (assumes Part 4, 6, 7 checkpoints exist) ---

# 1. Run the PPO training loop
python train_ppo.py --policy_ckpt../part_6/runs/sft-demo/model_last.pt                     --reward_ckpt../part_7/runs/rm-demo/model_last.pt                     --out runs/ppo-demo --steps 200 --batch_size 4 --resp_len 128                     --bpe_dir../part_4/runs/part4-demo/tokenizer

# 2. Evaluate the new PPO-aligned policy
python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt                    --reward_ckpt../part_7/runs/rm-demo/model_last.pt                    --bpe_dir../part_4/runs/part4-demo/tokenizer
Part 9: RLHF with Group-Relative Policy Optimization (GRPO)
This final part implements an alternative, simpler RLHF algorithm that avoids the need for a learned value head.

Conceptual Overview :   

9.1 Group-Relative Baseline: This method (train_grpo.py ) generates k (e.g., 4) completions for each prompt. It computes the mean reward of this group.   

9.2 Advantage Calculation: The advantage for each completion is its (reward - group mean reward). This group-relative baseline replaces the learned value network from Part 8, avoiding the instability of training a separate value head.   

9.3 Policy-Only Objective: The loss function (ppo_policy_only_losses ) is a PPO-style clipped objective that only updates the policy. It does not have a value loss component.   

9.4 KL Regularization: A simple, explicit KL penalty (KL(π 
policy
​
 ∣π 
ref
​
 )) is added directly to the loss  to keep the policy from diverging from the SFT model.   

This section highlights the curriculum's completeness by not just teaching one method (PPO), but by providing a modern, comparative alternative (GRPO), allowing users to explore different alignment techniques.

Key Files :   

File	Purpose
orchestrator.py	Runs unit tests and a tiny GRPO demo.
train_grpo.py	
(9.1) The main GRPO-RLHF training loop.

grpo_loss.py	
(9.3) The policy-only clipped objective + explicit KL penalty.

eval_ppo.py	
(Shared) Evaluates the final policy's reward.

policy.py	
(Shared) The PolicyWithValue model (value head is ignored).

rollout.py	
(Shared) Sampling and logprob utilities.

tests/test_grpo_loss.py	
Unit test for the GRPO loss.

  
Usage : All commands must be run from within the part_9/ directory. (Note: The orchestrator script  references running from part_8/, but the file structure implies part_9/. The commands below are corrected for the Part 9 context).   

Bash
# Navigate to the directory
cd part_9

# Run tests and the tiny GRPO demo
python orchestrator.py --demo

# --- Or run steps manually (assumes Part 4, 6, 7 checkpoints exist) ---

# 1. Run the GRPO training loop
python train_grpo.py --group_size 4                      --policy_ckpt../part_6/runs/sft-demo/model_last.pt                      --reward_ckpt../part_7/runs/rm-demo/model_last.pt                      --out runs/grpo-demo --steps 200 --batch_prompts 4 --resp_len 128                      --bpe_dir../part_4/runs/part4-demo/tokenizer

# 2. Evaluate the new GRPO-aligned policy
python eval_ppo.py --policy_ckpt runs/grpo-demo/model_last.pt                    --reward_ckpt../part_7/runs/rm-demo/model_last.pt                    --bpe_dir../part_4/runs/part4-demo/tokenizer
License
This project is licensed under the GNU General Public License v3 (GPLv3), as detailed in the LICENSE file.   

The GPLv3 is a "copyleft" license, which means that any derivative works or distributed software that incorporates this code must also be released under the GPLv3 license, ensuring the code remains free and open-source. As the license preamble states, "you must pass on to the recipients the same freedoms that you received... You must make sure that they, too, receive or can get the source code".   

Contributing
Contributions are welcome. This project follows a standard open-source workflow. Please feel free to:

Fork the repository.

Create a new feature branch (git checkout -b feature/MyAwesomeFeature).

Commit your changes (git commit -m 'Add MyAwesomeFeature').

Push to your branch (git push origin feature/MyAwesomeFeature).

Open a Pull Request.
