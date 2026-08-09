import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
lora_model_path = "./training/qwen1.5b-corrector-lora"
output_dir = "./training/qwen1.5b-corrector-lora-merged"

print(f"Loading base model {base_model_name} on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

print(f"Loading LoRA adapter from {lora_model_path}...")
model = PeftModel.from_pretrained(base_model, lora_model_path)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_dir}...")
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_dir)

print("LoRA adapter weights successfully merged into base model for production use!")
