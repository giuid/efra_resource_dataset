#%%
import os,gc
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"
os.environ['HF_CACHE_PATH'] = '/home/francomaria.nardini/raid/guidorocchietti/efra/data/.cache/huggingface'

from torch.utils.data.dataloader import DataLoader
from transformers import  AutoTokenizer
import torch
from huggingface_hub import login
from torch import nn
access_token_read = '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'
login(token = access_token_read)

from transformers import AutoModelForSeq2SeqLM
import json
import pandas as pd
from tqdm import tqdm


print('Loading data...')
with open('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/efra.json') as f:
    data = json.load(f)
efra_df = pd.DataFrame(data)
if os.path.exists('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/manual_parsed_df.csv') and False:
    manual_parsed_df = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/manual_parsed_df.csv')
else:
    parsed_html_path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/parsed_htmls/'
    parsed_files = os.listdir(parsed_html_path)
    df_dict = {}
    for file in tqdm(parsed_files[:]):
        with open(parsed_html_path + file, 'r') as f:
            html = json.load(f)
            for key in html.keys():
                df_dict[key] = " ".join(html[key]['texts'])
    parsed_df = pd.DataFrame(df_dict, index = [0]).T.reset_index()
    parsed_df.columns = ['post_id','text']


    manual_df = efra_df[efra_df['dataset'] == 'manual_summary']
    manual_ids = manual_df['post_id'].tolist()
    manual_parsed_df = parsed_df[parsed_df['post_id'].isin(manual_ids)]
    manual_parsed_df = manual_parsed_df.merge(manual_df[['post_id','english_summary']],  on = 'post_id')
    manual_parsed_df.to_csv('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/manual_parsed_df.csv', index = False)
# %%
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xxl', device_map='auto')
tokenized_texts_t5 = tokenizer(manual_parsed_df['text'].tolist(), padding='max_length', truncation=True, return_tensors='pt', max_length=512)
tokenized_summaries_t5 = tokenizer(manual_parsed_df['english_summary'].tolist(), padding='max_length', truncation=True, return_tensors='pt', max_length=512)
# %%
train_tokenized = tokenized_texts_t5['input_ids'][:int(len(tokenized_texts_t5['input_ids'])*0.8)]
train_labels = tokenized_summaries_t5['input_ids'][:int(len(tokenized_texts_t5['input_ids'])*0.8)]
validation_tokenized = tokenized_texts_t5['input_ids'][int(len(tokenized_texts_t5['input_ids'])*0.8):]
validation_labels = tokenized_summaries_t5['input_ids'][int(len(tokenized_texts_t5['input_ids'])*0.8):]

## Save tokenized data
torch.save(train_tokenized, '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/input_files/flan_t5/train_tokenized.pt')
torch.save(train_labels, '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/input_files/flan_t5/train_labels.pt')
torch.save(validation_tokenized, '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/input_files/flan_t5/validation_tokenized.pt')
torch.save(validation_labels, '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/input_files/flan_t5/validation_labels.pt')
#%%
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.targets[idx]
        }
train_dataset = SummarizationDataset(inputs=train_tokenized, targets=train_labels)
validation_dataset = SummarizationDataset(inputs=validation_tokenized, targets=validation_labels)

#%%
from accelerate import Accelerator
from transformers import AdamW
from torch.utils.data import DataLoader
from evaluate import load
from tqdm import tqdm
import os

# Initialize Accelerator
accelerator = Accelerator()

# Prepare DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=4)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Load ROUGE metric for evaluation
rouge = load("rouge")

# Prepare model, optimizer, and data for distributed training
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# Track the best metric
best_metric = None
save_dir = "/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/finetuned_models/flan_t5_xxl_manual_summary/best_model"  # Directory to save the best model
if  not(os.path.exists(save_dir)):
    os.makedirs(save_dir)
# Training loop
model.train()
epochs = 15
eval_steps = 100  # Evaluate every 500 steps
global_step = 0
early_stopping = 0
# Outer loop for epochs

for epoch in range(epochs):
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    
    for batch in epoch_bar:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        loss = outputs.loss

        # Backward pass
        accelerator.backward(loss)
        optimizer.step()

        global_step += 1

        # Update tqdm bar description with loss
        epoch_bar.set_postfix({"Loss": loss.item()})

        # Perform evaluation every `eval_steps`
        if global_step % eval_steps == 0:
            model.eval()
            all_preds = []
            all_labels = []

            val_bar = tqdm(val_loader, desc="Evaluating", leave=False)
            for val_batch in val_bar:
                with torch.no_grad():
                    # Generate predictions
                    generated_ids = model.generate(val_batch["input_ids"], max_length=128)

                    # Decode predictions and references
                    preds = accelerator.gather_for_metrics(
                        tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    )
                    labels = accelerator.gather_for_metrics(
                        tokenizer.batch_decode(val_batch["labels"], skip_special_tokens=True)
                    )

                    all_preds.extend(preds)
                    all_labels.extend(labels)

            # Compute metrics
            result = rouge.compute(predictions=all_preds, references=all_labels, use_stemmer=True)
            accelerator.print(f"Step {global_step}, Evaluation Metrics: {result}")

            # Check if this is the best model so far
            current_metric = result["rougeL"]#.mid.fmeasure  # Example: using ROUGE-L F1
            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                accelerator.print(f"New best model found at step {global_step}, saving model...")
                
                # Save the model and tokenizer
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping > 5:
                    break

            model.train()  # Switch back to training mode
# %%
