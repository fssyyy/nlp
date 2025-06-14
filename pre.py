import json
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# === Step 1: 表情与清洗 ===
def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    text = remove_emoji(text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9|，。！？、；：“”‘’《》…（）【】]", "", text)
    return text.strip()

# === Step 2: 加载模型 ===
model_path = "./hate_model/checkpoint-13500"  # ← 请修改为你保存的 checkpoint 目录
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# === Step 3: 加载测试数据 ===
with open("data/test1.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === Step 4: 生成预测 ===
results = []
for item in test_data:
    input_text = clean_text(item["content"])
    input_prompt = f"请从下面句子中抽取出仇恨主体、仇恨表达、类型和客体：{input_text}"
    inputs = tokenizer(input_prompt, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(prediction)

# === Step 5: 写入结果文件 ===
with open("data/predict2.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("预测完成，结果已保存至 predict.txt")
