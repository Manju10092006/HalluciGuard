import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenCorrectorClient:
    def __init__(self, model_path="./training/qwen1.5b-corrector-lora-merged"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            try:
                path = self.model_path
                if not os.path.exists(path):
                    path = "Qwen/Qwen2.5-1.5B-Instruct"
                    
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                
                # Setup device map - if CUDA available use it, else CPU
                device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=dtype, device_map=device_map
                )
                
                # Only load adapter if we loaded the base model
                if path == "Qwen/Qwen2.5-1.5B-Instruct":
                    adapter_path = "./training/qwen1.5b-corrector-lora"
                    if os.path.exists(adapter_path):
                        from peft import PeftModel
                        print(f"Loading LoRA adapter from {adapter_path}...")
                        self.model = PeftModel.from_pretrained(self.model, adapter_path)
                else:
                    print(f"Loaded merged model from {path} directly.")
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

    def generate_correction(self, prompt: str) -> str:
        self._load_model()
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        
        if hasattr(inputs, "keys"):
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            out = self.model.generate(
                **inputs, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True
            )
            input_len = inputs["input_ids"].shape[-1]
        else:
            # It's a single tensor
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            out = self.model.generate(
                inputs, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True
            )
            input_len = inputs.shape[-1]
            
        text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return text.strip()
