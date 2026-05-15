from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from dotenv import load_dotenv
import os
from argparse import ArgumentParser
from mdok import Mdok2Classifier
from datasets import Dataset
import json
import torch

model = None
tokenizer = None
classifier = None


def load_model(model_name, classifier_path):
    global model, tokenizer, classifier
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    classifier = Mdok2Classifier(classifier_path)


def reward_fn(completions, **kwargs):
    """Score completions: 1.0 = human-like, 0.0 = AI-like."""
    return [1.0 - classifier.predict(c) for c in completions]


def load_json_dataset(year):
    path = f"data/vk/task-vk-test-{year}.json"
    with open(path, "r") as f:
        return json.load(f)


def load_prompts(datasets):
    prompts = []
    for dataset in datasets:
        print(f"Loading dataset test-{dataset}...")
        data = load_json_dataset(dataset)
        task = data['voight-kampfftesttopics']
        prompt = task['prompt']
        topics = task['topics']
        for topic in topics:
            c_prompt = prompt + "\n" + topic['Content'] + "\n\n" + topic['Genre and Style'] + "\n\nThe text is: \n\n"
            chat = [
                {"role": "system", "content": "You are a helpful assistant that generates helpful answers. You will avoid pleasantries and small talk, focusing on the task at hand."},
                {"role": "system", "content": "You will avoid short paragraphs and bullet points."},
                {"role": "user", "content": c_prompt},
                {"role": "assistant", "content": ""},
            ]
            inputs = tokenizer.apply_chat_template(chat, tokenizer=tokenizer, return_tensors="pt")["input_ids"][0]  
            c_prompt = tokenizer.decode(inputs, skip_special_tokens=False)
            prompts.append(c_prompt)
    return prompts


if __name__ == "__main__":
    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-instruct")
    parser.add_argument("--classifier_path", type=str, default="mdok/classifier_model")
    parser.add_argument("--datasets", type=str, default="2026")
    parser.add_argument("--output_dir", type=str, default="./modelGRPO")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpointsGRPO")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_generations", type=int, default=8, help="Group size G: completions per prompt per step.")
    parser.add_argument("--max_completion_length", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=.10)
    args = parser.parse_args()

    torch.manual_seed(42)

    load_model(args.model_name, args.classifier_path)

    prompts = load_prompts(args.datasets.split(","))
    dataset = Dataset.from_list([{"prompt": p} for p in prompts])

    output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name.replace("/", "_"))

    grpo_config = GRPOConfig(
        output_dir=checkpoint_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations * 2,  # Generate more to have a good selection for GRPO
        max_completion_length=args.max_completion_length,
        save_strategy="no",
        logging_steps=10,
        beta=args.beta #new
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
