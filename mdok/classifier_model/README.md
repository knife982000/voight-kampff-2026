---
base_model: FacebookAI/roberta-large
library_name: peft
language:
- en
license: mit
pipeline_tag: text-classification
tags:
- base_model:adapter:FacebookAI/roberta-large
- lora
- transformers
- ai-detection
- human-vs-machine
- pan
- eloquent
- clef2026
---

# Mdok2 — AI-Generated Text Classifier

Binary sequence classifier that distinguishes **human-written** from **AI-generated** text. It is used as the reward model for GRPO fine-tuning of the Dargk team's submission to the [Voight-Kampff task](https://eloquent-lab.github.io/task-voight-kampff/) at [ELOQUENT Lab 2026](https://eloquent-lab.github.io/), CLEF 2026.

Mdok2 is inspired by, but distinct from, the original Mdok system. It is a LoRA fine-tuned version of [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large).

## Model Details

- **Developed by:** Dargk Team — Antonela Tommasel & Juan Manuel Rodriguez
- **Base model:** [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large) — 355M parameter encoder-only transformer
- **Model type:** Sequence classification — LoRA adapter (PEFT) with fully trained classification head
- **Language:** English
- **License:** MIT
- **Labels:** `0` = human-written, `1` = machine-generated

## Training

### Training data

The model was trained on the **PAN25 AI-generated text detection dataset (Task 1)** (`train.jsonl` + `val.jsonl`). Both splits were concatenated for training; the original validation split was used for evaluation. Duplicates and rows with missing text were removed.

### Preprocessing

Before tokenisation, all texts are:

1. Lowercased
2. Email addresses replaced with `[EMAIL]`
3. `@username` mentions replaced with `[USER]`
4. Phone numbers replaced with `[PHONE]`

Texts are truncated to **512 tokens**.

Training data also includes homoglyph augmentation: each character has a 5% probability of being replaced by a visually confusable Unicode character, and a 5% probability of being followed by a zero-width joiner (U+200D), to improve robustness against adversarial Unicode tricks.

### Training configuration

| Parameter | Value |
|---|---|
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | `query`, `value` |
| Modules trained fully | `classifier`, `pooler`, `score` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| LR schedule | Constant (with warmup) |
| Warmup steps | 3% of training steps |
| Batch size | 4 |
| Optimizer | paged_adamw_32bit |
| Loss | Class-weighted cross-entropy (balanced) |
| Max sequence length | 512 tokens |

```bash
python mdok-binary2.py \
    --train_file_path ../data/pan25-generative-ai-detection-task1-train/train.jsonl \
    --dev_file_path   ../data/pan25-generative-ai-detection-task1-train/val.jsonl \
    --model FacebookAI/roberta-large
```

### Evaluation metrics

AUC, Accuracy, Macro F1, and MAE are computed on the validation split.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re, torch

model_path = "Dargk/mdok2-roberta-large-ai-detector"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
model.eval()

email_re    = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
mention_re  = re.compile(r"@[A-Za-z0-9_-]+")
phone_re    = re.compile(r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}")

def preprocess(text):
    text = re.sub(email_re,   "[EMAIL]", text)
    text = re.sub(mention_re, "[USER]",  text)
    text = re.sub(phone_re,   " [PHONE]", text).replace("  [PHONE]", " [PHONE]")
    return text.lower().strip()

def predict(text):
    text = preprocess(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    p_machine = logits.softmax(dim=-1)[0, 1].item()
    return p_machine  # probability that the text is AI-generated

text = "The quick brown fox jumps over the lazy dog."
print(f"P(machine) = {predict(text):.3f}")
```

A score below **0.2** is treated as human-written in the Dargk GRPO pipeline.

## Role in the Dargk system

Mdok2 serves a dual purpose:

1. **Reward model during GRPO fine-tuning:** the reward for each generated completion is `1 − p(machine)`, pushing the LLM to produce more human-like text.
2. **Selector during inference:** up to 10 candidate texts are generated per prompt; the one with the lowest `p(machine)` is selected (or the first that falls below 0.2).

## Contact

**Dargk Team**

- Antonela Tommasel — antonela.tommasel@isistan.unicen.edu.ar
- Juan Manuel Rodriguez — jmro@cs.aau.dk

### Framework versions

- PEFT 0.18.1
