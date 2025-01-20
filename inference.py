#%%
import pandas as pd
import os, json
from huggingface_hub import login
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'

api_token =  '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'##'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'
login(token = api_token)
efra_summaries = load_dataset('giuid/efra_legal_dataset',  use_auth_token=api_token)
efra_summaries_df = pd.DataFrame(efra_summaries['train'])
# %%
htmls_path = '/home/francomaria.nardini/raid/guidorocchietti/code/efra_git/data/htmls/'
english_path = os.path.join(htmls_path, 'english')
original_path = os.path.join(htmls_path, 'original')
htmls =  {'post_id':[], 'english_html':[], 'original_html':[], 'summaries' : []}

for post_id in efra_summaries_df['post_id']:
    if os.path.exists(os.path.join(english_path, f'{post_id}.html')):
        with open(os.path.join(english_path, f'{post_id}.html'), 'r') as f:
            htmls['english_html'].append(f.read())
    if os.path.exists(os.path.join(original_path, f'{post_id}.html')):
        with open(os.path.join(original_path, f'{post_id}.html'), 'r') as f:
            htmls['original_html'].append(f.read())
    htmls['post_id'].append(post_id)
    htmls['summaries'].append(efra_summaries_df[efra_summaries_df['post_id']==post_id]['summary'])

# %%

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset

model_id = 'meta-llama/Llama-3.1-405B-Instruct-FP8'
model_id = 'meta-llama/Llama-3.3-70B-Instruct'
model_id = 'meta-llama/Llama-3.2-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model (use device_map="auto" for automatic placement)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",           # Automatically use FP8 or FP16 if supported
    device_map="auto",             # Automatically distribute layers across devices
    ### add token api key

)

# %%
