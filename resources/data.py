import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO
from html.parser import HTMLParser

from typing import Callable, Optional, Union, List
from transformers import PreTrainedTokenizer

#====================================================================================================#
# Data access and analytics:                                                                         #
#====================================================================================================#

from google.cloud import storage
from google.oauth2 import service_account

DATABASE    = 'c-labs1.efra.summaries_v3'
CREDENTIALS = service_account.Credentials.from_service_account_file('data/service-account-external-efra.json')

#====================================================================================================#
# Data processing:                                                                                   #
#====================================================================================================#

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

SENT_TOKENIZER = PunktSentenceTokenizer()

from html.parser import HTMLParser

class HTMLSplitter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = ''

    def __call__(self, html:str, window:int, tokenize:Callable[[str], List]=word_tokenize):
        parts, paragraphs, tokens, texts = [], [], [], []

        self.feed(html)
        html = self.get_data()

        # split text in paragraphs:
        part = 0
        cursor = 0
        remaining_tokens = -1
        for paragraph, txt in enumerate(html.split('\n\n')):
            for i,j in SENT_TOKENIZER.span_tokenize(txt):
                sentence = tokenize(txt[i:j+1])
                remaining_tokens -= len(sentence) - 1

                if remaining_tokens <= 0:
                    parts.append(str(part))
                    paragraphs.append(str(paragraph))
                    tokens.append(sentence[:window])
                    texts.append(txt[i:j+1])
                    remaining_tokens = window - len(sentence) + 1
                        
                    part += 1
                    cursor = i

                else: 
                    tokens[-1].extend(sentence[:window])
                    texts[-1] = txt[cursor:j+1]

            remaining_tokens = -1

        return {'part':parts, 'paragraph':paragraphs, 'tokens':tokens, 'texts':texts}

    @property
    def ends_with_space(self):
        if len(self.text) == 0: return True
        else: return self.text[-1] in (' ', '\n', '\r', '\t', '>')

    @property
    def ends_with_newline(self):
        if self.text.endswith('\n  - '): return True
        elif len(self.text) == 0: return True
        else: return self.text[-1] in ('\n', '\r', '>')
    
    def handle_starttag(self, tag, attrs):
        if tag in ('table', 'tr', 'th', 'td'):
            self.text += f'<{tag}>' if self.ends_with_newline else f'\n<{tag}>'
        
        elif tag == 'b' or tag == 'strong':
            self.text += '**' if self.ends_with_space else ' **'

        elif tag == 'i':
            if not self.ends_with_space:
                self.text += '*' if self.ends_with_space else ' *'
        
        elif tag == 'li':
            self.text += '  - ' if self.ends_with_newline else '\n  - '
        
        elif tag == 'p':
            if not self.ends_with_newline:
                self.text += '\n'
        
        elif tag.startswith('h'):
            self.text += '\n**' if self.ends_with_newline else '\n\n**'

        elif not self.ends_with_space:
            self.text += ' '

    def handle_endtag(self, tag):
        if tag in ('table', 'tr', 'th', 'td'):
            self.text += f'</{tag}>'
        
        if tag == 'b' or tag == 'strong':
            self.text += '** '

        elif tag == 'i':
            self.text += '* '
        
        elif tag == 'p':
            if not self.ends_with_space:
                self.text += '\n'
        
        elif tag.startswith('h'):
            self.text += '**\n'

        elif not self.ends_with_space:
            self.text += ' '
    
    def handle_data(self, data):
        self.text += data.replace('\n', '').replace('\r', '').strip()

    def get_data(self):
        return self.text.strip()

class Data:
    def __init__(self, url:str="hf://datasets/giuid/efra_legal_dataset/efra.csv"):
        self.data = pd.read_csv(url)

    def __getitem__(self, keys):
        # download data:
        data = self.data.loc[keys]

        # create bucket:
        client = storage.Client(credentials=CREDENTIALS)
        bucket = client.bucket('c-labs1-efra')

        for column in data.columns:
            if column.endswith('_url'):
                data[column[:-4]] = ['' if isinstance(url, float) else str(bucket.blob(url[18:]).download_as_string()) for url in tqdm(data[column].values, desc=f'Downloading column "{column[:-4]}"')]

        print(f'Retrieved data: {len(data):d} posts')

        return data

def save_summary(dir:str, df:pd.DataFrame, output_text:List[str], tokenizer:PreTrainedTokenizer, init:bool=False):
    #print('Saving...')
    save_df = df.copy()
    save_df['prompt'] = [tokenizer.decode(m) for m in save_df['prompt']]
    save_df['summary'] = output_text
    save_df.to_csv(os.path.join(dir,'summaries.csv'), mode='w' if init else 'a', index=False, header=init)

    for i in save_df.ID.drop_duplicates():
        txt = ''
    
        for s in save_df[save_df.ID == i]['summary'].values:
            s = s.split('**Summary:**')[-1]
            s = s.split('**Sentence:**')[-1]
            s = s.split('**Answer:**')[-1]
    
            txt += s
    
        with open(os.path.join(dir,f'summary_{i}.md'), 'w') as file:
            file.write(txt)

    del save_df
    gc.collect()