import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os




model = AutoModelForCausalLM.from_pretrained("Ottis/finnish-text-summarizer", access_token, torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("Ottis/finnish-text-summarizer", access_token)

def generate_summary(input_text):
    

    prompt = f"""<|alku|> Olet tekoälyavustaja. Tiivistä annettu teksti mahdollisimman kattavasti sekä tarkasti.
<|ihminen|> Tässä tiivistettävä teksti:
{input_text}
<|avustaja|> Vastauksesi:
"""
    print("Formatted Prompt:", prompt)
    input = tokenizer(prompt, return_tensors = "pt").to('cuda')
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input["input_ids"],
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|loppu|>"),
            attention_mask=input["attention_mask"],
            temperature=0.1,
            penalty_alpha = 0.6,
            top_k = 3,
            do_sample=True,
            repetition_penalty = 1.28,
            min_new_tokens = 25,
            max_new_tokens = 250
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    summary = generated_text.split('<|avustaja|> Vastauksesi:')[1]
    return summary







interface = gr.Interface(
    fn=generate_summary, 
    inputs="text", 
    outputs="text",
    title="Tekstin tiivistys",
    description="Teksti"
)

interface.launch()