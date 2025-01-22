import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from rouge_score import rouge_scorer
from longdocfactscore.ldfacts import LongDocFACTScore
from .data import HTMLSplitter
from .bart_score import BARTScorer
from nltk.tokenize import word_tokenize

from typing import List, Dict, Union

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TopicScorer:
    def __init__(self, originals:List[str], summaries:List[str], topics:List[List[str]]):
        super().__init__()

        all_topics = []
        for item in topics:
            for topic in item:
                if not topic in all_topics:
                    all_topics.append(topic)
        all_topics.sort()

        targets = np.array([[topic in ts for topic in all_topics] for ts in topics], dtype=int)
        print(targets.mean(axis=0))

        def fit_pipe(texts, targets):
            pipe = Pipeline([
                ('embedding',  TfidfVectorizer(tokenizer=word_tokenize)),
                ('classifier', OneVsRestClassifier(LogisticRegression(class_weight='balanced')))
            ])
            gs = GridSearchCV(pipe, {'classifier__estimator__penalty':['l1','l2'], 'classifier__estimator__C':[.1, 1., 10.]},
                              scoring='f1_macro', n_jobs=-1, refit=True)
            return gs.fit(texts, targets)

        self.pipeline_originals = fit_pipe(originals, targets)
        self.pipeline_summaries = fit_pipe(summaries, targets)

    def score(self, originals:List[str], summaries:List[str]):
        results = self.pipeline_originals.predict_proba(originals)
        results *= self.pipeline_summaries.predict_proba(summaries)
        print(results.mean(axis=0))
        return results.sum(axis=1)



class ParagraphScorer(HTMLSplitter):
    def __init__(self, score:str='rougeL', window:int=None, device:Union[str, torch.device, int]=DEVICE):
        super().__init__()

        if score.startswith('rouge'):
            self.window      = window if window is not None else 1024
            scorer           = rouge_scorer.RougeScorer([score], use_stemmer=True)
            self.scoring_fcn = lambda originals, summaries: [scorer.score(o, s)[score].fmeasure for o, s in zip(originals, summaries)]

        elif score == 'bart':
            self.window      = window if window is not None else 1024
            scorer           = BARTScorer(device=device, max_length=window)
            self.scoring_fcn = lambda originals, summaries: scorer.score(originals, summaries)

        else: raise ValueError(f'Unknown score "{score}"')

    def __call__(self, originals:List[str], summaries:List[str], **kwargs):
        return self.score(originals, summaries, **kwargs)

    def score(self, originals:List[str], summaries:List[str], **kwargs):
        for original, summary in zip(tqdm(originals), summaries):
            parts = super().__call__(original, self.window, **kwargs)
            parts['scores'] = self.scoring_fcn(parts['texts'], [summary]*len(parts['texts']))

            yield parts

        
        

def score(originals:List[str], summaries:List[str], topics:List[List[str]],
          rouge_types:List[str]=['rouge1', 'rougeL'], device:Union[str, torch.device, int]=DEVICE) -> Dict[str, float]:
    # just to be sure...
    assert len(originals) == len(summaries)
    assert len(originals) == len(topics)

    # get all topics:
    all_topics = []
    for item in topics:
        for topic in item:
            if not topic in all_topics:
                all_topics.append(topic)
    all_topics.sort()

    # compute scores:
    scores = {}

    # compute ROUGE-F1:
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    for key in rouge_types: scores[key] = []
    for o, s in zip(originals, summaries):
        rouge = scorer.score(o, s)
        for key in rouge: scores[key].append(rouge[key].fmeasure)

    # compute Long-Doc-Facts-Score:
    scorer = LongDocFACTScore(device=device)
    scores['ldfacts'] = scorer.score_src_hyp_long(originals, summaries)

#    # topic classification:
#    splitter = HTMLSplitter()
#    for #TODO