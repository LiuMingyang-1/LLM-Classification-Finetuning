import os
import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, load_from_disk
import shutil
import os
from datasets import load_from_disk, Dataset

# --- 配置 ---
# 建议先用 1.5B 快速跑通流程，确认无误再换 7B
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  
MAX_LENGTH = 2048
OUTPUT_DIR = "./qwen_kaggle_output"
CACHE_DIR = "./qwen_data_cache"



def prepare_dataset(csv_path, tokenizer, cache_dir="./processed_data_cache"):
    # --- 1. 尝试读取缓存 ---
    if os.path.exists(cache_dir):
        print(f"✨ 发现缓存目录 '{cache_dir}'，正在直接加载...")
        try:
            dataset = load_from_disk(cache_dir)
            print(f"✅ 加载成功！包含 {len(dataset['train'])} 条训练数据。")
            return dataset
        except Exception as e:
            print(f"⚠️ 缓存加载失败（可能是数据损坏），将重新处理。错误信息: {e}")

    # --- 2. 如果没缓存，开始处理数据 ---
    print("⚡ 未发现可用缓存，开始从头处理数据...")
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)

    # 标签映射
    def get_label(row):
        if row['winner_model_a'] == 1: return 0
        if row['winner_model_b'] == 1: return 1
        return 2 
    
    df['labels'] = df.apply(get_label, axis=1)
    
    # 构建 Prompt
    def construct_prompt(row):
        return (
            f"User Question: {row['prompt']}\n\n"
            f"Response A: {row['response_a']}\n\n"
            f"Response B: {row['response_b']}\n\n"
            f"Which response is better? Answer (Response A / Response B / Tie)."
        )
    
    df['text'] = df.apply(construct_prompt, axis=1)
    
    raw_dataset = Dataset.from_pandas(df[['text', 'labels']])
    
    def preprocess_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=MAX_LENGTH,
            padding=False 
        )
    
    print("🚀 正在 Tokenize (多进程加速中)...")
    tokenized_dataset = raw_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=["text"], # 必须删除 text 列
        num_proc=4               # 🔥【新增】开启4个进程并行处理，速度快4倍
    )
    
    # 划分数据集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # --- 3. 保存缓存 ---
    print(f"💾 正在保存数据到 '{cache_dir}'，下次运行将无需等待...")
    split_dataset.save_to_disk(cache_dir)
    print("✅ 保存完成！")
    
    return split_dataset

# --- 2. 评估 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main():
    # --- 加载 Tokenizer ---
    # Qwen 不需要 trust_remote_code=True (但在某些旧环境中加上也不报错)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Qwen 默认就有 pad_token (<|endoftext|> 或 <|im_end|>)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 准备数据 ---
    dataset = prepare_dataset("train.csv", tokenizer)
    
    # --- 加载模型 ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Qwen 完美兼容，不需要手动设置 config.pad_token_id，但为了保险：
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # --- LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],    
        
        )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # =========== ✨ 核心修复代码开始 ✨ ===========
    # 强制将所有"可训练参数"转为 float32
    # 这一步能解决 "Attempting to unscale FP16 gradients" 报错
    # 同时也提高了 LoRA 训练的数值稳定性
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    # =========== ✨ 核心修复代码结束 ✨ ===========

        # 4. Trainer 参数微调
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4, # LoRA 学习率通常比全量微调大，但 2e-4 可能略大，1e-4 或 5e-5 比较稳
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=8, # 累计步数增加，等效 Batch Size = 2*8 = 16，更稳定
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,      # 步数不用太频繁，省时间
        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        fp16=True,
        report_to="none",
        label_names=["labels"],
        warmup_ratio=0.03,   # 增加预热，防止刚开始梯度爆炸
        metric_for_best_model="eval_loss", # 以 loss 为准
        greater_is_better=False,           # loss 越小越好
        load_best_model_at_end=True        # 训练结束加载最好的模型
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()