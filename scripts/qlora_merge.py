from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch



base_model_path = ''
peft_model_path = ''
output_dir = ''

def main(base_model_path, peft_model_path, output_dir):
    device = "cuda"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.bfloat16
    ).to(device)  

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, peft_model_path).to(device)
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)



main(base_model_path, peft_model_path, output_dir)