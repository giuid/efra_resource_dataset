
#%%
import  os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"  # Set the CUDA devices to use
os.environ['HF_HOME'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'

import argparse


#%%

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Embed passages using a specified model.")
    parser.add_argument("--model_name", type=str, default="facebook/dragon-plus-context-encoder", help="Name of the model to use.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the tokenizer.")
    #parser.add_argument("--index_type", type=str, default="passages", help="Type of index to generate. ['passages', 'documents']")
    parser.add_argument("--dataset_path", type=str, default="./data/", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default="lepard_passages.csv", help="Name of the dataset file.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for embedding.")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Interval for saving checkpoints.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save embeddings.")    
    parser.add_argument("--cuda_devices", type=str, default="0,1", help="CUDA devices to use.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from tqdm.auto import tqdm
    from resources.src import embed_passages
    
    model_name = args.model_name
    max_length = args.max_length
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    checkpoint_interval = args.checkpoint_interval
    save_dir = args.save_dir or os.path.join(dataset_path, 'passage_embeddings', model_name.replace('/', '_'))
    #index_type = args.index_type
    #if index_type.lower() == 'passage': index_type = 'passages' 
    #elif index_type.lower() == 'document': index_type = 'documents'
    #assert index_type in ['passages', 'documents'], "Invalid index type. Choose from ['passages', 'documents']"
    
    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map='auto')
    model.eval()

    print('Loading dataset...')
    collection = pd.read_csv(os.path.join(dataset_path, dataset_name),on_bad_lines='warn', names=['id', 'text'], header=0)
    collection = collection.dropna()
    os.makedirs(save_dir, exist_ok=True)
    embeddings_file = os.path.join(save_dir, 'passage_embeddings.npy')
    passages = collection['text'].tolist()
    embeddings = []

    print('Checking for existing checkpoint...')
    if os.path.exists(embeddings_file):
        print(f"Checkpoint found: {embeddings_file}")
        embeddings = np.load(embeddings_file)
        start_index = embeddings.shape[0]
        embeddings = list(embeddings)
        print(f"Resuming from passage {start_index}")
    else:
        print("No checkpoint found. Starting from scratch.")
        embeddings = []
        start_index = 0

    print('Embedding passages...')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for step, i in enumerate(tqdm(range(start_index, len(passages), batch_size), desc="Embedding passages"), start=1):
        batch = passages[i:i + batch_size]
        batch_embeddings = embed_passages(batch, model, tokenizer, device=device, max_length=max_length)
        embeddings.append(batch_embeddings)

        if step % checkpoint_interval == 0:
            np.save(embeddings_file, np.vstack(embeddings))
            print(f"Checkpoint saved at step {step}: {embeddings_file}")

    embeddings = np.vstack(embeddings)
    np.save(embeddings_file, embeddings)

    print(f"Embeddings saved to: {embeddings_file}")
    
    
# Example call for the script
# python /home/francomaria.nardini/raid/guidorocchietti/code/efficiency_retrieval/index_trec_generic.py --model_name "Snowflake/snowflake-arctic-embed-l-v2.0" --max_length 512 --dataset_path "../../data/conversational/CAST2019/" --dataset_name "CASTcollection.tsv" --batch_size 2048 --checkpoint_inwterval 100 --save_dir "/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/snowflake-arctic-embed-l-v2.0" --cuda_devices "1,2,4"
### select the passages file and the model name == 'facebook/dragon-plus-context-encoder' and the max_length = 512 and the batch_size = 2048 and the checkpoint_interval = 100 and the save_dir = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/lepard_embeddings/passages/dragon-plus-context-encoder' and the cuda_devices = 3
# python /home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/generate_embeddings.py --model_name "facebook/dragon-plus-context-encoder" --max_length 512 --dataset_path "/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/datasets/" --dataset_name "lepard_passages.csv" --batch_size 2048 --checkpoint_interval 100 --save_dir "/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/lepard_embeddings/passages/dragon-plus-context-encoder" --cuda_devices 3

# python /home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/generate_embeddings.py --model_name "Snowflake/snowflake-arctic-embed-l-v2.0" --max_length 512 --dataset_path "./data/datasets" --dataset_name "lepard_passages.csv" --batch_size 512 --checkpoint_inwterval 100 --save_dir "/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/lepard_embeddings/passages/snowflake-arctic-embed-l-v2.0" --cuda_devices "3,6"
