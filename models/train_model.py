from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd



train_path = "/"
valid_path = "/"


train_dataset = load_dataset("csv", data_files={'train': train_path}, split='train')
valid_dataset = load_dataset("csv", data_files={'test': valid_path}, split='test')


# Quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)



model = AutoModelForCausalLM.from_pretrained("Finnish-NLP/llama-7b-finnish", quantization_config=quant_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/llama-7b-finnish")

# Add custom tokens to manage structured responses and align model embeddings.
# Source: https://huggingface.co/Finnish-NLP/llama-7b-finnish-instruct-v0.1/blob/main/train_unsloth_7b.py
tokenizer.clean_up_tokenization_spaces=True
tokenizer.add_tokens(["<|alku|>", "<PAD>", "<|ihminen|>", "<|avustaja|>"])
tokenizer.pad_token = "<PAD>"
tokenizer.add_special_tokens({'eos_token': '<|loppu|>'})
tokenizer.add_tokens('\n', special_tokens=True)
tokenizer.add_eos_token=True
model.resize_token_embeddings(new_num_tokens=len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id



response_template = "\n<|avustaja|> Vastauksesi:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)


collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)


# lora params

peft_parameters = LoraConfig(
    lora_alpha=128,
    target_modules = ["q_proj","v_proj"],
    lora_dropout=0.1,
    r=128,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save = ["lm_head", "embed_tokens"],
    use_rslora = True
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_parameters)


OUTPUT_DIR = ""

training_arguments = TrainingArguments(
    warmup_steps = 50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    logging_steps=40,
    learning_rate=2e-5,
    fp16=True,
    bf16=False,
    num_train_epochs=2,
    save_strategy="steps",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    weight_decay = 0.001,
    max_steps=-1,
    seed=42,
    evaluation_strategy="steps",
    save_steps = 40
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_parameters,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=collator

)

trainer.train()