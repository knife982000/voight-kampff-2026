from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from peft import PeftModel
import re

class Mdok2Classifier:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, device_map="auto")
        self.email_pattern = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")                          # e.g., name@example.com
        self.user_mention_pattern = re.compile(r"@[A-Za-z0-9_-]+")                                                 # e.g., @my_username
        self.phone_pattern = re.compile(r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}") #modified from https://stackabuse.com/python-regular-expressions-validate-phone-numbers/
        
    def preprocess(self, text):
        text = re.sub(self.email_pattern, "[EMAIL]", text)
        text = re.sub(self.user_mention_pattern, "[USER]", text)
        text = re.sub(self.phone_pattern, " [PHONE]", text).replace('  [PHONE]', ' [PHONE]')
        return text.lower().strip()

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=-1)[0, 1].item()  # Assuming binary classification, get the probability of the positive class
        return probabilities

    def predict(self, text):
        return self.classify(self.preprocess(text))
    

if __name__ == "__main__":
    classifier = Mdok2Classifier("mdok/classifier_model")
    import pandas as pd
    from tqdm import tqdm
    val = pd.read_json("data/pan25-generative-ai-detection-task1-train/val.jsonl", lines=True)
    res = []
    true = []
    for _, row in tqdm(val.iterrows(), total=len(val)):
        text = row["text"]
        label = row["label"]
        pred = classifier.classify(text)
        res.append(pred)
        true.append(label)
    from sklearn.metrics import classification_report
    print(classification_report(true, res))