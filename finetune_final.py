"""
Covent LLM Challenge - Fine-tuning script
Qwen2.5-1.5B-Instruct + unsloth + QLoRA, on a free-tier Colab/Kaggle T4 GPU.

This is meant to be run cell-by-cell in a notebook (Colab or Kaggle), not
as a single python script - split it at each blank-line-separated section
below into its own cell. Requires train.jsonl (from generate_dataset.py)
in the working directory.

Before running, install dependencies:
    pip install --no-deps unsloth unsloth_zoo "trl>=0.18.2,<=0.24.0" peft accelerate bitsandbytes
    pip install -q transformers==4.57.6 --force-reinstall --no-deps
    pip install -q "huggingface-hub>=0.34.0,<1.0" --force-reinstall --no-deps
    pip install -q hf_transfer
"""

# --- load the base model with QLoRA ---
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN = 512  # records + prompt are short, this keeps training fast

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,  # let it pick automatically
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- build the train/val split and format as chat examples ---
import json
import random
from datasets import Dataset

SYSTEM_PROMPT = (
    "You normalize messy US property records into a fixed JSON schema. "
    "Given a raw record, output ONLY valid JSON with exactly these keys: "
    "parcel_id, situs_address, situs_city, situs_state, situs_zip, "
    "owner_name, owner_mailing_address, owner_mailing_city, "
    "owner_mailing_state, owner_mailing_zip. "
    "If a value is unknown, use null. Do not add commentary."
)


def load_pairs(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


raw_pairs = load_pairs("train.jsonl")
random.Random(42).shuffle(raw_pairs)
n_val = max(50, int(len(raw_pairs) * 0.05))
val_pairs = raw_pairs[:n_val]
train_pairs_split = raw_pairs[n_val:]
print(f"Train: {len(train_pairs_split)}  Val: {len(val_pairs)}")


def to_chat_text(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Normalize this record:\n{example['input']}"},
        {"role": "assistant", "content": json.dumps(example["target"], ensure_ascii=False)},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}


train_ds = Dataset.from_list(train_pairs_split).map(to_chat_text)
val_ds = Dataset.from_list(val_pairs).map(to_chat_text)
print(train_ds[0]["text"][:500])

# --- train ---
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=20,
        num_train_epochs=2,  # cut down from 3 - validation loss was already
                             # plateauing by epoch 2.3 in an earlier run
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()

# --- quick sanity check on a validation example ---
FastLanguageModel.for_inference(model)


def normalize(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Normalize this record:\n{text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(inputs, max_new_tokens=200, temperature=0.1, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


test_input = val_pairs[0]["input"]
print("INPUT:\n", test_input)
print("\nMODEL OUTPUT:\n", normalize(test_input))
print("\nEXPECTED:\n", json.dumps(val_pairs[0]["target"], indent=2))

# --- save the model ---
model.save_pretrained("qwen25-1.5b-parcel-normalizer")
tokenizer.save_pretrained("qwen25-1.5b-parcel-normalizer")
print("Saved.")
