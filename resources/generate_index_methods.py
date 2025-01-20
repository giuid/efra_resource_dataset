#%%
#transformers.cache_dir = '/home/francomaria.nardini/raid/guidorocchietti/.cache'
import os, json
import numpy as np
from tqdm.auto import tqdm
os.environ['HF_HOME'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # Set the CUDA devices to use
import torch
from transformers import AutoTokenizer, AutoModel
import faiss


# %%


### Define function to get texts and ids from parsed files (json) ###
### Every file contains a dictionary with keys as the query ids and values as the texts ###
### The key is the file name of the original HTML from where the passages where extracted ###
def get_texts(files_path):
    texts = []
    ids = []
    parsed_files = os.listdir(files_path)
    for file in tqdm(parsed_files):
        with open(files_path + file, 'r') as f:
            html = json.load(f)
            for key in html.keys():
                texts += html[key]['texts']
                ids += ([key]*len(html[key]['texts']))
    return texts, ids

#%%            
### Define function to embed a list of passages using a model and a tokenizer ###
def embed_passages(passages, model, tokenizer, device="cuda", max_length=512):
    inputs = tokenizer(passages, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings =outputs.last_hidden_state[:, 0, :]  # Mean pooling
    return embeddings.cpu().numpy()
### Define function to generate the context embeddings used to later build the index ###
### The embeddings are saved in a folder as numpy arrays ###
### The function takes a list of texts, the model name and the output folder as input ###
### The function uses the last hidden state of the model to generate the embeddings ###
### The function uses a step parameter to avoid memory issues when generating the embeddings ###
### The function uses the first token of the last hidden state as the embedding for the context ###
def generate_context_embeddings(texts,model_name = 'facebook/dragon-plus-query-encoder',output_folder = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/passages_tensors/', step = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    context_encoder = AutoModel.from_pretrained(model_name, device_map='auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context_encoder.to(device)
    for i in tqdm(range(0, len(texts), step)):
        tkns = tokenizer(texts[i:i+step], padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        tkns = {k: v.to(device) for k, v in tkns.items()}
        embeddings = context_encoder(**tkns).last_hidden_state[:, 0, :]
        ## Save embeddings
        embeddings = embeddings.detach().cpu().numpy()
        with open(f'{output_folder}{i}.npy', 'wb') as f:
            np.save(f, embeddings)

# %%
### Define function to load the embeddings from a folder ###
### The function loads all the numpy arrays in the folder and stacks them ###

def load_embeddings_from_folder(path):
    embeddings = None
    files = os.listdir(path)
    for file in tqdm(files[:]):
        with open(path + file, 'rb') as f:
            if embeddings is None:
                embeddings = np.load(f)
            else:
                embeddings = np.vstack((embeddings, np.load(f)))
    return embeddings

#%%
### Define function to create a FAISS index from a list of embeddings ###
### The function takes a numpy array of embeddings and an optional save_folder parameter ###
### The function creates a FAISS index and adds the embeddings to it ###
### The function saves the index to disk if a save_folder is provided ###

def create_faiss_index(embeddings, save_folder = None):
    # Create a FAISS index
    num_vectors = len(embeddings)
    dim = len(embeddings[0])
    faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    # Add vectors to the FAISS index
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    if save_folder is not None:
        print(f'Saving index to {save_folder}')
        faiss.write_index(faiss_index, save_folder)
    return faiss_index