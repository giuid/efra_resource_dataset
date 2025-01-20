#%%
import pandas as pd
import os, json
from huggingface_hub import login
from datasets import load_dataset
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES']="3,4,5,6,7"

from resources.data import word_tokenize, SENT_TOKENIZER, HTMLSplitter
from typing import Callable, List, Dict

def split_text_in_paragraphs(
    html: str, 
    window: int = 512, 
    tokenize: Callable[[str], List[str]] = word_tokenize
) -> Dict[str, List]:
    """
    Splits HTML text into paragraphs and tokenizes them.

    Args:
        html (str): The HTML content to process.
        window (int): Maximum number of tokens per part.
        tokenize (Callable[[str], List[str]]): Function to tokenize text.

    Returns:
        dict: A dictionary with parts, paragraphs, tokens, and texts.
    """
    parts, paragraphs, tokens, texts = [], [], [], []
    parser = HTMLSplitter()
    parser.feed(html)
    html = parser.get_data()

    part, cursor, remaining_tokens = 0, 0, -1
    for paragraph_idx, paragraph_text in enumerate(html.split('\n\n')):
        for start, end in SENT_TOKENIZER.span_tokenize(paragraph_text):
            sentence_tokens = tokenize(paragraph_text[start:end + 1])
            remaining_tokens -= len(sentence_tokens) - 1

            if remaining_tokens <= 0:
                parts.append(str(part))
                paragraphs.append(str(paragraph_idx))
                tokens.append(sentence_tokens[:window])
                texts.append(paragraph_text[start:end + 1])
                remaining_tokens = window - len(sentence_tokens) + 1
                part += 1
                cursor = start
            else:
                tokens[-1].extend(sentence_tokens[:window])
                texts[-1] = paragraph_text[cursor:end + 1]

        remaining_tokens = -1

    return " ".join(texts),{'part': parts, 'paragraph': paragraphs, 'tokens': tokens, 'texts': texts}

def prepare_input(texts,tokenizer,instruction = "", assistant = "", max_input_length = 1024):
    tokenized_instruction = tokenizer.tokenize(instruction)
    tokenized_assistant = tokenizer.tokenize(assistant)

    prompts = []
    for txt in texts:
        max_text_tokens = max_input_length - len(tokenized_instruction) - len(tokenized_assistant)
        tokenized_txt = tokenizer.tokenize(txt)
        truncated_text_tokens = tokenized_txt[:max_text_tokens]
        whole_text_tokenized = tokenized_instruction + truncated_text_tokens + tokenized_assistant
        prompt = tokenizer.convert_tokens_to_string(whole_text_tokenized)
        prompts.append(prompt)
    inputs = tokenizer(
        prompts,
        max_length=max_input_length,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )
    return prompts, inputs

def generate_summaries(model, inputs, tokenizer, max_output_length=256, num_beams=4):
    """
    Generates summaries using the model.

    Args:
        model: Pre-trained summarization model.
        inputs: Tokenized input tensors.
        tokenizer: Pre-trained tokenizer.
        max_output_length (int): Maximum length of output summaries.
        num_beams (int): Beam search width for generation.

    Returns:
        List[str]: Generated summaries as a list of strings.
    """
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_output_length,
            num_beams=num_beams,
            early_stopping=True
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
# %%
api_token =  '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'##'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'
login(token = api_token)
efra_summaries = load_dataset('giuid/efra_legal_dataset',  use_auth_token=api_token)
efra_summaries_df = pd.DataFrame(efra_summaries['train'])

htmls_path = '/home/francomaria.nardini/raid/guidorocchietti/code/efra_git/data/htmls/'
english_path = os.path.join(htmls_path, 'english')
original_path = os.path.join(htmls_path, 'original')
htmls =  {'post_id':[], 'english_html':[], 'original_html':[], 'summaries' : [], 'english_paragraphs':[], 'original_paragraphs':[]}

efra_summaries_df_subset = efra_summaries_df[efra_summaries_df['post_id'].isin([x.split('.html')[0] for x in os.listdir(english_path)])]

for post_id in efra_summaries_df_subset['post_id']:
    if os.path.exists(os.path.join(english_path, f'{post_id}.html')):
        with open(os.path.join(english_path, f'{post_id}.html'), 'r') as f:
            txt, parag = split_text_in_paragraphs(f.read())
            htmls['english_html'].append(txt)
            htmls['english_paragraphs'].append(parag)
    if os.path.exists(os.path.join(original_path, f'{post_id}.html')):
        with open(os.path.join(original_path, f'{post_id}.html'), 'r') as f:
            txt, parag = split_text_in_paragraphs(f.read())
            htmls['original_html'].append(txt)
            htmls['original_paragraphs'].append(parag)
    htmls['post_id'].append(post_id)
    htmls['summaries'].append(efra_summaries_df_subset[efra_summaries_df_subset['post_id']==post_id]['english_summary'])


# %%

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
import torch

model_id = 'meta-llama/Llama-3.1-405B-Instruct-FP8'
model_id = 'meta-llama/Llama-3.3-70B-Instruct'
model_id = 'meta-llama/Llama-3.2-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model (use device_map="auto" for automatic placement)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",           # Automatically use FP8 or FP16 if supported
    device_map="auto",             # Automatically distribute layers across devices
)

# %%
model.eval() 
max_input_length = 1024
texts = htmls['english_html']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer.pad_token = tokenizer.eos_token 

inputs = tokenizer(
        texts,
        max_length=max_input_length,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )
tokenized_inputs = {key: value.to(device) for key, value in inputs.items()}





cut = 3000


max_input_length = 1024
instruction = f"### User:\n I will provide you with a legal document regarding food and risk regulation. I want you to generate a summary of the document. Text: " 
assistant = f"\n### Summary:\n"


                                                 
#%%               
                                  
prompts, tokenized_inputs = prepare_input(texts,tokenizer,instruction = instruction, assistant = assistant, max_input_length = max_input_length)
inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
summaries = generate_summaries(model, inputs, tokenizer, max_output_length=cut, num_beams=4)
# %%
def clean_output(x):
    splitted = x.split('### User:\n')
    answers = [x.split('### Summary:\n')[-1] for x in splitted]
    last_dot = ['.'.join(x.split('.')[:-1]) + '.' for x in answers]
    return ''.join(last_dot)[1:]
cleaned_summaries = [clean_output(x) for x in summaries]
print(cleaned_summaries)
# %%
