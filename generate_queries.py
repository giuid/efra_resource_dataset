
# %%
import os
os.environ['HF_HOME'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,6'
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
access_token_read = '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'
login(token = access_token_read)

lepard = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/datasets/lepard_dataset.csv')
#%%
# Load Llama model and tokenizer
model_id = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',torch_dtype=torch.float16)

#%%

# Legal text
legal_text =lepard.document.iloc[0]

# Prompt
prompt = f"Given the following legal text:\n{legal_text}\nGenerate specific, relevant questions that can be asked to a search engine to retrieve relevant documents."
messages = [
    {"role": "system", "content": "You are a user that poses queries to a legal search engine. I will provide a text from which you need to generate a query. Answer only with the query."},
    {"role": "user", "content": f"{legal_text}"},
]
# Generate queries
inputs = tokenizer(messages, return_tensors="pt").to("cuda:0")
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda:0")

attention_mask = torch.ones_like(input_ids)  # Create a mask of ones
inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}
#%%
def prepare_inputs(legal_text):
    messages = [
        {"role": "system", "content": "You are a user that poses queries to a legal search engine. I will provide a text from which you need to generate a query. Answer only with the query."},
        {"role": "user", "content": legal_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        padding="max_length",
        max_length=4096,
        truncation=True,
        return_tensors="pt"
    ).to("cuda:0")
    attention_mask = torch.ones_like(input_ids)  # Create a mask of ones
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return inputs
all_inputs = [prepare_inputs(legal_text) for legal_text in lepard.document.tolist()[:4]]
batched_input_ids = torch.cat([inputs["input_ids"] for inputs in all_inputs], dim=0).to("cuda:0")
batched_attention_masks = torch.cat([inputs["attention_mask"] for inputs in all_inputs], dim=0).to("cuda:0")
batched_inputs = {
    "input_ids": batched_input_ids,
    "attention_mask": batched_attention_masks,
}

#%%
outputs = model.generate(**batched_inputs, max_new_tokens=128)
generated_text = [tokenizer.decode(el, skip_special_tokens=True) for el in outputs]
prompt_text = [tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True) for inputs in all_inputs]
#prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

# Get the new text by slicing out the prompt
new_generated_text = generated_text[len(prompt_text):].strip()
new_generated_text = [generated[len(prompt):].strip() for generated,prompt in zip(generated_text,prompt_text)]

print(new_generated_text)
# %%
generatex = []
for i in range(4):
    inputs= prepare_inputs(lepard.unique().document.iloc[i])
    outputs = model.generate(**inputs, max_new_tokens=128)
    tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    new_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt_text):].strip()
    generatex.append(new_generated_text)
# %%
