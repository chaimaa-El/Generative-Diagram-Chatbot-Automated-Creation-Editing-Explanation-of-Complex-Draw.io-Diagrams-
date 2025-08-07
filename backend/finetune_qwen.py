import torch
from unsloth import FastLanguageModel
from prepare_dataset import load_data
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

#Load and standardize dataset
dataset = load_data("data/cleaned_dataset.json")
dataset = standardize_sharegpt(dataset)

#Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.03,
    bias="none",
    use_gradient_checkpointing=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    random_state=42,
    use_rslora=True,
    loftq_config=None,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

#Train-validation split
df = dataset.to_pandas()
train_df, eval_df = train_test_split(df, test_size=0.20, stratify=df["intent"], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

#Format prompts
train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=eval_dataset.column_names)

#Tokenize the text field and set labels
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

#Training arguments
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=10,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,
    warmup_steps=20,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    seed=42,
)

#Initialize and run the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
)

trainer.train()

#Save final model and tokenizer
model.save_pretrained("qwen2.5-lora-diagram-chatbot-final")
tokenizer.save_pretrained("qwen2.5-lora-diagram-chatbot-final")