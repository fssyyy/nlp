import json
import random
import re
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)

# === Step 0: 表情处理 ===
def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# === Step 1: 文本清洗与格式统一 ===
def clean_text(text):
    text = remove_emoji(text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9|，。！？、；：“”‘’《》…（）【】]", "", text)
    return text.strip()

def normalize_output(output):
    output = output.replace("｜", "|").replace(" - ", " | ")
    output = re.sub(r"\s*\|\s*", " | ", output)
    output = re.sub(r"\s*\[SEP\]\s*", " [SEP] ", output)
    return output.strip()

# === Step 2: 可选数据增强 ===
def synonym_augment(text, prob=0.3):
    SYNONYMS = {
        "被歧视": ["遭到偏见", "遭遇歧视", "被排斥"],
        "人人喊打": ["大家喊打", "所有人讨厌"],
        "碰瓷": ["提一下", "蹭一蹭", "提及"],
        "基佬": ["同性恋", "同志"],
    }
    for key, syns in SYNONYMS.items():
        if key in text and random.random() < prob:
            text = text.replace(key, random.choice(syns))
    return text

# === Step 3: 加载原始数据 + 增强 + 划分 ===
def load_and_split_data(input_path, train_ratio=0.9, augment=False):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if augment:
        augmented = []
        for item in data:
            augmented.append(item)
            new_item = item.copy()
            new_item["content"] = synonym_augment(item["content"])
            augmented.append(new_item)
        data = augmented

    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

# === Step 4: 构建tokenizer和模型 ===
model_name = "IDEA-CCNL/Randeng-T5-784M"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# === Step 5: 预处理函数 ===
def preprocess_function(example):
    cleaned_input = clean_text(example["content"])
    cleaned_output = normalize_output(example["output"])
    input_text = f"请从下面句子中抽取出仇恨主体、仇恨表达、类型和客体：{cleaned_input}"
    target_text = cleaned_output
    model_input = tokenizer(input_text, max_length=128, padding="max_length", truncation=True)
    label = tokenizer(target_text, max_length=128, padding="max_length", truncation=True)
    model_input["labels"] = label["input_ids"]
    return model_input

# === Step 6: 加载并处理数据集 ===
train_data, valid_data = load_and_split_data("data/train.json", augment=True)
train_dataset = Dataset.from_list(train_data).map(preprocess_function)
valid_dataset = Dataset.from_list(valid_data).map(preprocess_function)

# === Step 7: 设置训练参数 ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./hate_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    fp16=False
)

# === Step 8: 构建Trainer并训练 ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint=True)

