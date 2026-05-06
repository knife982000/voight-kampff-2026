CACHE=None

from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, BitsAndBytesConfig, Phi3ForSequenceClassification
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import datetime
import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig, PeftModel, AutoPeftModelForCausalLM, TaskType, AutoPeftModelForSequenceClassification, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch
import torch.nn.functional as F
from accelerate import PartialState
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
import re
from confusables import confusable_characters
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def report_gpu():
    pass

H_PROBABILITY = 0.05
ZWJ_PROBABILITY = 0.05

def homoglyph(input_text):
    output = ""
    for char in input_text:
      if (random.random() < H_PROBABILITY):
        output += confusable_characters(char)[int(random.random()*len(confusable_characters(char)))]
      else:
        output += char
      if (random.random() < ZWJ_PROBABILITY):
        output += "\u200D"
    return output

def preprocess(text):
  EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")                          # e.g., name@example.com
  USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")                                                 # e.g., @my_username
  PHONE_PATTERN = re.compile(r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}") #modified from https://stackabuse.com/python-regular-expressions-validate-phone-numbers/
  text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
  text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
  text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace('  [PHONE]', ' [PHONE]')
  return text.lower().strip()

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=512)


def get_data(train_path, dev_path, random_seed):
    """
    function to read dataframe with columns
    """
    
    train = pd.read_json(train_path, lines=True)
    valid = pd.read_json(dev_path, lines=True)
    
    train = pd.concat([train, valid])
    train.dropna(subset=['text'], inplace=True)
    train.drop_duplicates(subset=['text'], inplace=True, ignore_index=True)
    
    train['text'] = [preprocess(x) for x in train['text']]
    valid['text'] = [preprocess(x) for x in valid['text']]

    return train, valid


def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    probs = predictions[:,1]
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results = {"AUC": roc_auc_score(labels, probs), "ACC": accuracy_score(labels, predictions), "MacroF1": f1_score(labels, predictions, average='macro'), "MAE": mean_absolute_error(labels, predictions)}
    return results

pos_weights = 1.0
neg_weights = 1.0
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "base_layer" in lora_module_names:  # problem with training from peft checkpoint
        lora_module_names.remove("base_layer")
    return list(lora_module_names)

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model, continue_train):
    #class weghts
    global CACHE
    global pos_weights
    global neg_weights
    pos_weights = len(train_df) / (2 * len(train_df[train_df.label == 1]))
    neg_weights = len(train_df) / (2 * len(train_df[train_df.label == 0]))

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    floatorbfloat = torch.float32 #torch.float16
    if 'lama' in model:
        floatorbfloat = torch.bfloat16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=floatorbfloat,
        bnb_4bit_quant_storage=floatorbfloat,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_name = model
    if 'bert' in model_name.lower():
      bnb_config=None
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=floatorbfloat,
        #attn_implementation="flash_attention_2", # use sdpa, alternatively use "flash_attention_2"
        cache_dir=CACHE,
        trust_remote_code=True,
        num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.config.use_cache = False
    
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    
    target_modules=[]
    if 'falcon' in model_name.lower():
      target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif 'mistral' in model_name.lower():
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    #elif 'llama' in model_name:
    #  pass #target_modules=['v_proj', 'q_proj', 'k_proj', 'o_proj'] #'down_proj', 'up_proj', 'gate_proj', #['v_proj', 'up_proj', 'gate_proj', 'o_proj', 'down_proj', 'k_proj', 'q_proj']
    elif 'bert' in model_name.lower():
      target_modules=["query_proj", "key_proj", "value_proj"]
    else:
      target_modules=find_all_linear_names(model)
    
    modules_to_save=["score"]
    if 'bert' in model_name.lower(): modules_to_save=["classifier", "pooler"]
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        #target_modules=target_modules, #"all-linear", #
        modules_to_save=modules_to_save
    )
    print(model.state_dict().keys())
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    output_dir = checkpoints_path
    per_device_train_batch_size = 4#16 #4
    gradient_accumulation_steps = 1#4
    optim = "paged_adamw_32bit" #"adamw_torch" #"paged_adamw_8bit" #
    logging_steps = 1000 #10
    save_steps = logging_steps# * 5 #10
    learning_rate = 2e-5 #2e-4
    num_train_epochs=3
    warmup_steps = int(0.03 * len(train_df)) // per_device_train_batch_size // gradient_accumulation_steps
    lr_scheduler_type = "constant"
    # fp16 = False
    # bf16 = True
    # tf32 = False
    # if 'bert' in model_name.lower():
    #   tf32=True
    #   bf16=False
    # if 'lama' in model_name.lower():
    #     fp16 = False
    #     bf16 = True
                
    if continue_train:
      output_dir = model_name.split('/checkpoint-')[0]
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        #fp16=fp16,
        #bf16=bf16,
        #tf32=tf32,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        load_best_model_at_end=True,
        eval_strategy="steps",
    )
    
    max_seq_length = 512

    trainer = CustomTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        args=training_arguments,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
            
    if continue_train:
      trainer.train(resume_from_checkpoint=True)
    else:
      trainer.train()

    # save best model
    best_model_path = output_dir+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
        
    trainer.save_model(best_model_path)
    trainer.model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    for module in modules_to_save:
      try:
        torch.save(getattr(trainer.model,module).state_dict(), f'{best_model_path}/{module}-params.pt')
      except:
        print(f"Module {module} not dumped.")
    
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--dev_file_path", "-d", required=True, help="Path to the dev file.", type=str)
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument('--continue_train', '-c', action='store_true')
    parser.add_argument('--random_seed', '-rs', help="Set random seed to affect dataset shuffling.", type=int, default=42)
    parser.add_argument('--test_only', '-to', action='store_true')

    args = parser.parse_args()

    train_path =  args.train_file_path # For example 'subtaskA_train_multilingual.jsonl'
    dev_path =  args.dev_file_path
    model =  args.model # For example 'xlm-roberta-base'
    random_seed = args.random_seed #For example 1000
    

    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
    
    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df = get_data(train_path, dev_path, random_seed)
    
    # train detector model
    if args.test_only != True:
        fine_tune(train_df, valid_df, f"models/{model}", id2label, label2id, model, args.continue_train)
