#!/usr/bin/env python
# coding: utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login
from argparse import ArgumentParser
import os
import torch
import json
import pandas as pd
from tqdm.auto import tqdm

from mdok import Mdok2Classifier

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

parser = ArgumentParser()
parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-instruct")
parser.add_argument("--finetuned_model", type=str, default=None,
                    help="Path to a PEFT adapter to load on top of the base model.")
parser.add_argument("--output", type=str, required=True, help="Output CSV file path.")
parser.add_argument("--datasets", type=str, default="2024,2025,2026",
                    help="Dataset years, comma-separated (e.g. '2024,2025,2026').")
parser.add_argument("--classifier_path", type=str, default="mdok/classifier_model",
                    help="Path to the mdok classifier model.")
parser.add_argument("--max_tries", type=int, default=10,
                    help="Max generation attempts per prompt before falling back to best.")
parser.add_argument("--max_new_tokens", type=int, default=600)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_p", type=float, default=0.9)
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.base_model)
tokenizer.pad_token = tokenizer.eos_token

classifier = Mdok2Classifier(args.classifier_path)


def load_json_dataset(year):
    path = f"data/vk/task-vk-test-{year}.json"
    with open(path, "r") as f:
        return json.load(f)


def load_prompts(datasets):
    prompts, ids = [], []
    for dataset in datasets:
        print(f"Loading dataset test-{dataset}...")
        data = load_json_dataset(dataset)
        task = data['voight-kampfftesttopics']
        prompt = task['prompt']
        topics = task['topics']
        for topic in topics:
            prompts.append(
                prompt + "\n" + topic['Content'] + "\n\n"
                + topic['Genre and Style'] + "\n\nThe text is: \n\n"
            )
            ids.append(topic['id'])
    return ids, prompts


def generate_sample(prompt, model, seed):
    torch.manual_seed(seed)
    chat = [
        {"role": "system", "content": "You are a helpful assistant that generates helpful answers. "
                                      "You will avoid pleasantries and small talk, focusing on the task at hand."},
        {"role": "system", "content": "You will avoid short paragraphs and bullet points."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=1,
    )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return texts[0][len(decoded_prompt):]


ids, prompts = load_prompts(args.datasets.split(","))

model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="cpu",
    )
if args.finetuned_model is not None:
    print(f"Loading fine-tuned adapter from {args.finetuned_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model,
        torch_dtype="auto",
        device_map="cpu",
    )

model = model.to("cuda")

model = model.to("cuda")
model.eval()

out = []
with torch.no_grad():
    for topic_id, prompt in tqdm(zip(ids, prompts), total=len(ids)):
        best_text = None
        best_score = float("inf")  # lowest AI probability wins
        accepted = False

        for attempt in range(args.max_tries):
            text = generate_sample(prompt, model, seed=attempt)
            ai_score = classifier.predict(text)  # probability of being AI-generated

            if ai_score < best_score:
                best_score = ai_score
                best_text = text

            if ai_score < 0.2:  # classified as human (label 0)
                accepted = True
                break

        if not accepted:
            tqdm.write(
                f"[{topic_id}] No human-like text found in {args.max_tries} tries; "
                f"using best (AI score={best_score:.3f})"
            )

        out.append((topic_id, best_text))

df = pd.DataFrame(data=out, columns=["id", "text"])
df.to_csv(args.output, index=False)
print(f"Saved {len(out)} samples to {args.output}")
