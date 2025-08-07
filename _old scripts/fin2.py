import torch
from unsloth import FastLanguageModel
from prepare_dataset import load_data
from trl import SFTTrainer
from transformers import TrainingArguments,DataCollatorForSeq2Seq

from transformers import EarlyStoppingCallback, TrainerCallback
import matplotlib.pyplot as plt

from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = load_data("data/dataset.json")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",  
    max_seq_length=2048,
    dtype = None ,
    load_in_4bit=True,
    
)


model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    target_modules = ["q_proj", "k_proj", "v_proj"] ,
    random_state = 42,
    use_rslora = True ,
    loftq_config = None,
    
)

from unsloth.chat_templates import get_chat_template, standardize_sharegpt
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
dataset = standardize_sharegpt(dataset)



def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }


class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(step)
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10,6))
        plt.plot(self.steps, self.train_losses, label="Train Loss")
        if len(self.eval_losses) > 0:
            eval_steps = self.steps[:len(self.eval_losses)]
            plt.plot(eval_steps, self.eval_losses, label="Eval Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title("Training & Evaluation Loss")
        plt.savefig("loss_plot.png")
        plt.show()


dataset = dataset.shuffle(seed=42)
df = dataset.to_pandas()

train_df, eval_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["intent"],
    random_state=42,
)


train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)


train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

eval_dataset.save_to_disk("data/eval_dataset")




trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),

    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=8,                    
        max_steps=-1, 
                                 
        logging_steps=5,
        warmup_steps=80,
        
        save_strategy="steps",       
        save_steps=5,    
        max_grad_norm=0.5,   
        output_dir="outputs",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        eval_strategy="steps",
        eval_steps=5,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,


    ),
)


trainer.add_callback(LossPlotCallback())
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)


trainer.train()


model.save_pretrained("qwen2.5-lora-diagram-chatbotfinal2")
tokenizer.save_pretrained("qwen2.5-lora-diagram-chatbotfinal2")

# training_args = TrainingArguments(
#     output_dir="./qwen-finetuned",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     learning_rate=2e-4,
#     num_train_epochs=10,  
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     optim="paged_adamw_8bit",
#     warmup_steps=10,
# )

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="output",
#     max_seq_length=2048,
#     args=training_args,
# pip install torch transformers datasets pandas scikit-learn unsloth trl accelerate nltk rouge_score evaluate peft bitsandbytes absl-py



