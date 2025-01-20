#%%
from resources.generate_index_methods import *

files_path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/parsed_htmls/'
texts, ids = get_texts(files_path)

output_path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/passages_tensors/'
generate_context_embeddings(texts, output_folder = output_path)
# %%
embeddings = load_embeddings_from_folder(output_path)
np.save('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/passage_embeddings.npy', embeddings)
# %%
create_faiss_index(embeddings, '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/faiss_index_passages.index')
