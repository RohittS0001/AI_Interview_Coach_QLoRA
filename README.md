# AI Interview Coach — QLoRA

A parameter-efficient fine-tuning project that adapts **Qwen2.5-1.5B-Instruct** into an AI/ML interview-focused assistant using **LoRA and QLoRA**.

The project demonstrates the complete workflow from dataset preparation and supervised fine-tuning to evaluation and optimized inference using continuous batching.

## 🚀 Project Overview

Preparing for AI/ML interviews often requires repeatedly practicing technical questions across topics such as:

* Machine Learning
* NLP
* Transformers
* Large Language Models
* RAG
* Vector Databases
* LangChain
* Agentic AI
* ML Engineering

Instead of using a general-purpose language model, this project explores how a small instruction-tuned model can be adapted to an **AI Engineer interview domain** using parameter-efficient fine-tuning.

The model is fine-tuned using **QLoRA**, which combines:

* 4-bit NF4 quantization
* LoRA adapters
* Parameter-efficient supervised fine-tuning
* FP16 computation for quantized layers

The project also evaluates inference performance using **continuous batching and paged KV-cache management**.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Fine-tune a small LLM for AI/ML interview-related questions.
2. Understand and implement **LoRA**.
3. Understand and implement **QLoRA**.
4. Reduce GPU memory requirements through 4-bit quantization.
5. Evaluate the fine-tuned model against the base model.
6. Benchmark inference performance.
7. Explore continuous batching for handling multiple requests efficiently.

---

## 🧠 Model

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`

The model contains approximately **1.55B logical parameters** and is loaded using 4-bit NF4 quantization during QLoRA fine-tuning.

### Why Qwen2.5-1.5B?

A relatively small model was selected because the project is designed to demonstrate parameter-efficient fine-tuning on limited GPU resources rather than relying on a large multi-billion-parameter model.

---

# 🔧 LoRA

LoRA (Low-Rank Adaptation) allows a pretrained model to be adapted by training a small set of additional parameters instead of updating the entire model.

### LoRA configuration

| Parameter      |                                  Value |
| -------------- | -------------------------------------: |
| Rank (`r`)     |                                     16 |
| LoRA alpha     |                                     32 |
| Dropout        |                                   0.05 |
| Bias           |                                   None |
| Task           |               Causal Language Modeling |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

Only approximately **4.36M parameters** are trainable:

**4,358,144 / 1,548,072,448 = 0.2815%**

This significantly reduces the number of parameters that need to be updated during fine-tuning.

---

# ⚡ QLoRA

QLoRA combines LoRA with low-bit quantization.

The base model is loaded using:

* **4-bit quantization**
* **NF4 (NormalFloat 4-bit)**
* **Double quantization**
* **FP16 compute**

Conceptually:

```text
Qwen2.5-1.5B
       │
       ▼
4-bit NF4 Quantization
       │
       ▼
Frozen Base Model
       │
       +───────────────+
       │               │
       ▼               ▼
   LoRA Adapters   Forward Pass
       │               │
       │            FP16 Compute
       │               │
       +───────┬───────+
               ▼
             Loss
               │
               ▼
       Update LoRA Weights
```

The important distinction is that **quantization reduces the memory required to store the base model, while LoRA reduces the number of trainable parameters.**

---

# 📚 Dataset

**Dataset:** `raghu298/ml-interview-sft-dataset`

The original dataset contained **566 examples**.

After removing duplicate examples:

* Original examples: **566**
* Unique examples: **531**
* Duplicates removed: **35**

The dataset contains interview questions covering multiple AI/ML topics, including:

* Vector Databases
* Transformers
* LLMs
* ML Engineering
* NLP
* LangChain
* Agentic AI
* RAG
* and other AI/ML concepts

The data is formatted using the model's chat template before supervised fine-tuning.

---

# 🏋️ Training

The model is trained using **Supervised Fine-Tuning (SFT)**.

### Training setup

| Configuration           |           Value |
| ----------------------- | --------------: |
| Epochs                  |               3 |
| Learning rate           |          `2e-4` |
| Training batch size     |               4 |
| Gradient accumulation   |               2 |
| Effective batch size    |               8 |
| Maximum sequence length |             512 |
| Quantization            |       4-bit NF4 |
| Compute dtype           |            FP16 |
| GPU                     | NVIDIA Tesla T4 |

The effective training batch size is:

```text
4 × 2 = 8
```

Gradient accumulation allows a larger effective batch size without requiring the full batch to fit into GPU memory simultaneously.

---

# 📊 Evaluation

The fine-tuned model was evaluated using several complementary approaches.

## Training Metrics

| Metric          | Result |
| --------------- | -----: |
| Training loss   | 1.3445 |
| Validation loss | 1.4061 |
| Token accuracy  | 70.69% |
| Perplexity      |   4.08 |

The relatively small difference between training and validation loss suggests that the model did not exhibit a large train/validation gap on this dataset.

### Perplexity

Perplexity was calculated from the test loss:

```text
Perplexity = exp(Test Loss)
           ≈ exp(1.4061)
           ≈ 4.08
```

Perplexity is most meaningful when comparing models under the same evaluation setup.

---

# 📝 Base Model vs QLoRA

ROUGE-L was also used as an auxiliary metric to compare generated answers with reference answers.

| Model             | ROUGE-L |
| ----------------- | ------: |
| Base Qwen2.5-1.5B |  0.1213 |
| QLoRA model       |  0.2419 |

The QLoRA model achieved higher lexical overlap with the reference answers on this evaluation set.

However, ROUGE-L alone does **not** measure whether an open-ended interview answer is technically correct. Therefore, qualitative evaluation was also performed.

The comparison showed that the fine-tuned model learned interview-related patterns and terminology, but it could still produce technically incorrect or incomplete answers on some questions.

This highlights an important limitation of fine-tuning: **better training/evaluation metrics do not automatically guarantee factual or technical correctness.**

---

# ⚙️ Inference Optimization

The project also explores inference optimization using **continuous batching**.

Instead of processing requests one at a time:

```text
Request 1 → Model
Request 2 → Model
Request 3 → Model
Request 4 → Model
```

continuous batching allows multiple requests to be managed together:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼──► Continuous Batching ──► GPU
Request 4 ─┤
Request 5 ─┘
```

The implementation uses the Transformers continuous batching API with **paged KV-cache management**.

This allows the GPU to process multiple generation requests more efficiently and improves aggregate throughput.

---

# 🚀 Inference Benchmark

On an NVIDIA Tesla T4, the stable continuous-batching configuration achieved:

| Metric                |                Result |
| --------------------- | --------------------: |
| Requests processed    |                 1,000 |
| Total processing time |              ~421 sec |
| Throughput            | **2.37 requests/sec** |
| Peak VRAM             |           **7.48 GB** |

The benchmark dataset was created by repeating the available test examples to create a larger inference workload.

**Note:** These 1,000 requests are a stress-test workload and do not represent 1,000 unique interview questions.

---

# 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL
* bitsandbytes
* Accelerate
* Qwen2.5-1.5B-Instruct
* LoRA
* QLoRA
* NF4 Quantization
* CUDA
* Google Colab
* Continuous Batching
* Paged KV-Cache

---

# 📓 Project Structure

This project is intentionally provided as a single Jupyter notebook:

```text
AI_Interview_Coach_QLoRA/
│
└── AI_Interview_Coach_QLoRA.ipynb
```

The notebook contains the complete workflow:

```text
Environment Setup
       ↓
Dataset Loading
       ↓
Data Cleaning
       ↓
Train / Validation / Test Split
       ↓
Chat Template Formatting
       ↓
Base Model Loading
       ↓
LoRA Configuration
       ↓
QLoRA 4-bit Quantization
       ↓
Supervised Fine-Tuning
       ↓
Evaluation
       ↓
Base vs QLoRA Comparison
       ↓
Inference Benchmarking
       ↓
Continuous Batching
```

---

# 💻 Hardware

The training and inference experiments were performed using:

**GPU:** NVIDIA Tesla T4
**VRAM:** ~15 GB

The project was designed around a relatively constrained GPU environment to demonstrate that parameter-efficient fine-tuning can be performed without requiring a high-end GPU.

---

# ⚠️ Limitations

This project is primarily an **experimental and educational fine-tuning project**, rather than a production-ready interview platform.

Current limitations include:

* Small training dataset
* Small base model
* Limited evaluation set
* ROUGE-L does not measure technical correctness
* The model can still hallucinate or produce incomplete answers
* The benchmark workload contains repeated test examples
* Continuous-batching throughput does not represent individual request latency
* The current notebook does not implement a complete user-facing web application

Future improvements could include:

* Larger and higher-quality interview datasets
* Human evaluation
* LLM-as-a-judge evaluation
* Topic-specific scoring
* Answer correctness and relevance metrics
* Adaptive difficulty
* Follow-up interview questions
* Personalized feedback
* Voice-based interview practice
* Web application deployment
* Multi-turn interview sessions

---

# 🔮 Future Vision

The eventual goal is to evolve this model into a complete AI Interview Coach:

```text
Generate Interview Question
            ↓
        User Answer
            ↓
     Answer Evaluation
            ↓
   Score + Technical Feedback
            ↓
    Identify Weak Areas
            ↓
    Generate Follow-up
            ↓
     Final Interview Report
```

This would combine parameter-efficient fine-tuning with an interactive agentic interview workflow.

---

# 📌 Key Takeaways

This project demonstrates practical experience with:

* Parameter-efficient fine-tuning
* LoRA
* QLoRA
* 4-bit NF4 quantization
* Hugging Face PEFT
* Supervised fine-tuning
* LLM evaluation
* Model comparison
* GPU memory optimization
* Continuous batching
* Paged KV-cache management
* Inference benchmarking

---

## 👨‍💻 Author

**Rohit Sasane**

BCA — Artificial Intelligence & Machine Learning

Interested in:

**AI Engineering • Machine Learning • Generative AI • LLMs • RAG • Agentic AI**

---

## ⭐ If you find this project useful

Feel free to explore the notebook, experiment with the LoRA/QLoRA configuration, and build on the interview-coaching workflow.
