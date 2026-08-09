import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

import argparse

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test on 20 samples')
    args_cmd = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # We use bf16 if supported, else float16 or float32 depending on GPU capabilities.
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = None
    if device == "cuda":
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                MODEL, 
                quantization_config=quantization_config,
                device_map={"": 0}, 
                torch_dtype=dtype
            )
            model = prepare_model_for_kbit_training(model)
            model.config.use_cache = False
        except Exception as e:
            print(f"CUDA/4-bit loading failed: {e}")
            print("Falling back to CPU standard loading (this will use system RAM and be slower)...")
            device = "cpu"

    if device == "cpu":
        dtype = torch.float32 # Most compatible for CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, 
            device_map={"": "cpu"}, 
            torch_dtype=dtype
        )

    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files={
        "train": "training/data/train.jsonl",
        "validation": "training/data/val.jsonl"
    })
    
    if args_cmd.smoke_test:
        print("Running smoke test on first 20 records...")
        dataset["train"] = dataset["train"].select(range(min(20, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(5, len(dataset["validation"]))))

    def format_example(ex):
        # Huggingface's chat template directly expects a list of dicts with 'role' and 'content', which matches our jsonl exactly.
        return tokenizer.apply_chat_template(ex["messages"], tokenize=False)

    args = SFTConfig(
        output_dir="./training/qwen1.5b-corrector-lora-smoke" if args_cmd.smoke_test else "./training/qwen1.5b-corrector-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if args_cmd.smoke_test else 16,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        num_train_epochs=1 if args_cmd.smoke_test else 3,
        learning_rate=2e-4,
        logging_steps=1 if args_cmd.smoke_test else 10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        max_length=1500, # Ensure long prompts fit
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=format_example
    )

    trainer.train()
    
    out_dir = "./training/qwen1.5b-corrector-lora-smoke" if args_cmd.smoke_test else "./training/qwen1.5b-corrector-lora"
    trainer.save_model(out_dir)
    
    print(f"Training complete. Weights saved to {out_dir}")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM usage: {peak_vram:.2f} GB")

if __name__ == "__main__":
    train()
