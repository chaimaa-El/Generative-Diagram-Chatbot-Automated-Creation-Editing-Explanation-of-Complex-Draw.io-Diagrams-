import torch
from unsloth import FastLanguageModel
from prepare_dataset import load_data
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

# Load and process your data as before
dataset = load_data("data/cleaned_dataset.json")
dataset = standardize_sharegpt(dataset)
df = dataset.to_pandas()
train_df, eval_df = train_test_split(df, test_size=0.20, stratify=df["intent"], random_state=42)
eval_dataset = Dataset.from_pandas(eval_df)

# Save to disk
eval_dataset.save_to_disk("data/eval_dataset")