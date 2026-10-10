<div align="center">

# Hybrid Reasoning Agents

**LLM Agents that Think in latent space, Act through text**

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-models%20%26%20data-yellow.svg)](#-released-checkpoints-and-data)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Jiaru Zou](https://scholar.google.com/citations?user=GzLTey4AAAAJ)<sup>1,2</sup>, [Yunzhe Qi](https://scholar.google.com/citations?user=Gt17_A0AAAAJ)<sup>3,4</sup>, [Sirui Chen](https://csr16.github.io/)<sup>3</sup>, [Federico Bianchi](https://federicobianchi.io/)<sup>2</sup>, [Shang Zhu](https://shang-zhu.github.io/)<sup>2</sup>, [Pan Lu](https://lupantech.github.io/)<sup>1</sup>, [Jingrui He](https://www.hejingrui.org/)<sup>3</sup>, [James Zou](https://www.james-zou.com/)<sup>1,2</sup>

<sup>1</sup>Stanford University · <sup>2</sup>Together AI · <sup>3</sup>UIUC · <sup>4</sup>Meta

</div>

<p align="center"><img src="assets/hybrid_agent_overview.png" width="100%" alt="HybridAgent architecture overview"></p>

**HybridAgent** is an LLM agent that dynamically navigates between latent reasoning and text-based interaction within a single generation process. A lightweight **HybridLink** maps last-layer hidden states back into the input-embedding space, so the agent can enter **latent mode** for fast, inspectable inner thinking and return to **text mode** for tool calls, final answers and external monitoring. The complete hybrid context of text, latent thoughts and tool observations is carried across turns through the agent's own **working memory**, and every latent thought can be rendered as readable pseudo-text.

This repository contains the method: the hybrid generation engine with Python and search tools, the **HybridSFT** and **HybridRL** post-training stages, batch inference, and the **Interpretable Latent Mode** inspector.

## ⚙️ Installation

```bash
git clone https://github.com/togethercomputer/HybridAgent.git && cd HybridAgent
conda create -n hybridagent python=3.11 -y && conda activate hybridagent
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # TAVILY_API_KEY for the search tool
```

## 📁 Repository

| Directory | Contents |
|---|---|
| [`hybrid_agent/`](hybrid_agent/) | HybridLink, the model wrapper and loader, the hybrid generation engine (text/latent modes, working memory, concurrent tool calls) and the tools |
| [`hybrid_sft/`](hybrid_sft/) | HybridSFT: data format, objective and trainer |
| [`hybrid_rl/`](hybrid_rl/) | HybridRL: hybrid policy rollout, reward, exact rollout replay, policy update |
| [`hybrid_inference/`](hybrid_inference/) | Batch hybrid inference over a JSONL of questions |
| [`interpretable_latent_mode/`](interpretable_latent_mode/) | Latent thoughts as pseudo-text: HTML reports and terminal view |
| [`scripts/`](scripts/) | Launchers for the three stages |

## 🤗 Released models and data

Checkpoints and data load from a local directory or directly from the Hugging Face Hub.

| Artifact | Description | Hugging Face |
|---|---|---|
| HybridAgent-8B-Agentic | Qwen3-8B-Base → HybridSFT → HybridRL on agentic reasoning tasks | `togethercomputer/HybridAgent-8B-Agentic` |
| HybridAgent-14B-Agentic | Qwen3-14B-Base → HybridSFT → HybridRL on agentic reasoning tasks | `togethercomputer/HybridAgent-14B-Agentic` |
| HybridAgent-HybridSFT-v1 / -v2 | HybridSFT training sets: hybrid trajectories (`train.jsonl`), v1 and the upgraded v2 | `togethercomputer/HybridAgent-HybridSFT-v1`, `togethercomputer/HybridAgent-HybridSFT-v2` |
| HybridAgent-HybridRL | HybridRL training sets: deep reasoning (`math.parquet`) and agentic search (`search.parquet`) | `togethercomputer/HybridAgent-HybridRL` |

*Note:* Each model directory holds the backbone in Hugging Face format together with the HybridLink weights (`adapter.pt`, `adapter_config.json`).

## 🚀 Quick start

### Batch Hybrid Inference

```bash
python -m hybrid_inference.run_batch --model togethercomputer/HybridAgent-8B-Agentic \
    --input examples/questions.jsonl --output outputs/results.jsonl --num_samples 4
bash scripts/run_batch_inference.sh 8 --model togethercomputer/HybridAgent-8B-Agentic \
    --input questions.jsonl --output outputs/results.jsonl        # one process per GPU
```

One record per question: the hybrid trajectory, the extracted answer, the pseudo-text readout of every latent span and the tool-call statistics. See [`hybrid_inference/`](hybrid_inference/).

### Interpretable latent mode

```bash
python -m interpretable_latent_mode --model togethercomputer/HybridAgent-14B-Agentic \
    --question "How many trailing zeros does 100! have?" --output report.html
```

Renders the latent thoughts as pseudo-text inline with the text reasoning, tool calls and observations. HybridAgent-14B-Agentic on the first question of `examples/questions.jsonl` (terminal view, text segments omitted):

```text
QUESTION  At the speed of sound in air at 20 degrees Celsius, how many seconds does sound take
          to travel 10 kilometers? Round to the nearest second.

LATENT    [span 1 · 42 steps · readability 0.84]
          Wait, I me the the speed of speed in air at 20° is about 343 meters per second. But I
          should verify that. Let me do a search search to be sure.
SEARCH    speed of sound in air at 20 degrees celsius meters per second
RESULT    Page 1: The speed of sound in dry air at 20 °C is 343 meters per second. ...

LATENT    [span 2 · 22 steps · readability 0.88]
          Now, I need to calculate the time it takes for speed to travel 10 km at this speed.
LATENT    [span 3 · 21 steps · readability 0.96]
          So, time = 10,000 meters / 343 meters per second.
PYTHON    ... print(f"Rounded to the nearest second: {rounded_time} seconds")
RESULT    Time taken: 29.154518950437318 seconds
          Rounded to the nearest second: 29 seconds

LATENT    [span 7 · 10 steps · readability 0.92]
          So, the answer is 29 seconds.
ANSWER    \boxed{29}
```

See [`interpretable_latent_mode/`](interpretable_latent_mode/) for the full trace and the HTML report.

### HybridSFT

```bash
MODEL=Qwen/Qwen3-8B-Base DATA=togethercomputer/HybridAgent-HybridSFT-v2 OUT=outputs/hybrid_sft \
    bash scripts/run_hybrid_sft.sh all
```

Runs the three HybridSFT stages in sequence: format activation, HybridLink activation and self-conditioned HybridSFT. Recommended hardware: 8 or 16 H100 80 GB GPUs; the provided configs are for one node with 8 GPUs, and on 16 GPUs across two nodes `--grad_accum 1` keeps the same effective batch of 16. See [`hybrid_sft/`](hybrid_sft/).

### HybridRL

```bash
TASK=math MODEL=outputs/hybrid_sft/self_conditioned DATA=hf://togethercomputer/HybridAgent-HybridRL/math.parquet \
    NNODES=4 NODE_RANK=<node> MASTER_ADDR=<rank-0 host> bash scripts/run_hybrid_rl.sh
```

Trains HybridRL on the deep-reasoning tasks; `TASK=search` switches to the agentic search setting. Recommended hardware: 32 H100 80 GB GPUs (4 nodes × 8), the launcher default; the rollout batch is split across ranks, so `--prompts_per_step` must be divisible by the number of GPUs. See [`hybrid_rl/`](hybrid_rl/).

## 📚 Citation

```bibtex
@article{zou2026hybrid,
  title   = {Hybrid Reasoning Agents},
  author  = {Zou, Jiaru and Qi, Yunzhe and Chen, Sirui and Bianchi, Federico and Zhu, Shang and
             Lu, Pan and He, Jingrui and Zou, James},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```