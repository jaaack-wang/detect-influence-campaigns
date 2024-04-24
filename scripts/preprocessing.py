import re
import spacy
import numpy as np
from time import time
from tqdm import tqdm
import pandas as pd
import torch


def load_spacy(spacy_model):
    global nlp
    nlp = spacy.load(spacy_model, disable=["tok2vec", "tagger", 
                                            "parser", "attribute_ruler", 
                                            "lemmatizer", "ner"])
    nlp.add_pipe('sentencizer')


def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def preprocess(text):
    # should be defined according to actual needs 
    text = re.sub(r"(http\S+|www\S+|\S*//t.co/\S+)", "", text, flags=re.IGNORECASE) # url
    text = remove_emojis(text) # remove emojis
    text = re.sub(r"\S{21,}", "", text)
#     text = re.sub(r"[_@#]", "", text) # remove [-, @, #]
    text = re.sub(r"\s+", " ", text).strip() # redundant spaces
    return text


def get_sentences(text, lower_bound, upper_bound, 
                  max_sent_num, use_gpu=False):
    out = []
    if use_gpu:
        with torch.no_grad():
            sents = nlp(text).sents
    else:
        sents = nlp(text).sents
    
    if max_sent_num is not None:
        assert isinstance(max_sent_num, int), "max_sent_num must be int!"
        sents = list(sents)[:max_sent_num]
    
    for s in sents:
        # ignore sentence whose average token length is above 20 chars per word
        if (lower_bound <= len(s) <= upper_bound) and (len(s.text) / len(s) <= 20):
            out.append(s.text.strip())
    return out


def create_preprocessed_dataset(df, 
                                fp=None, 
                                break_sents=True,
                                lower_bound=5, 
                                upper_bound=100,
                                max_sent_num=None,
                                use_gpu=False, 
                                show_processing_info=True, 
                                spacy_model="en_core_web_lg"):
    
    '''df is a dataframe that must start with a `text` columns. The rest columns
       will be kept, so please make sure they are relevant to you.'''
    
    print(f"{'#'*20} Preprocessing data {'#'*20}\n")
    
    if show_processing_info:
        start = time()
    
    load_spacy(spacy_model)
    assert "text" in df.columns, "the input data must have a column named as 'text'!"
    
    if df.columns[0] != "text":
        new_cols = ["text"] + [c for c in df.columns if c != "text"]
        df = df[new_cols]
    
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(preprocess)
    
    cols = ["docID", "sentID"] + df.columns.to_list()
    texts_and_more = []
    
    if break_sents:
        break_text = lambda t: get_sentences(t, lower_bound, upper_bound,
                                             max_sent_num, use_gpu)
    else:
        def break_text(text):
            length = len(text.split())
            # ignore sentence whose average token length is above 20 chars per word
            if (lower_bound <= length <= upper_bound) and (len(text) / length <= 20):
                return [text]
            if length >= lower_bound:
                return [" ".join(text.split()[:upper_bound])]
            return []

    for i in tqdm(df.index):
        doc_ix = i
        sub = df.loc[i].to_list()
        text, rest = sub[0], sub[1:]
        
        ts = break_text(text)
        if ts:
            sent_ix = 1
            for t in ts:
                texts_and_more.append([doc_ix, sent_ix, t] + rest)
                sent_ix += 1
    
    new_df = pd.DataFrame(texts_and_more, columns=cols)
    
    if show_processing_info:
        end = time()
        print("Processing time (sec):", end-start)
        
        doc_num = new_df.docID.unique().size
        print("Num of texts:", len(new_df), "Num of docs:", doc_num)    
        
        texts = new_df["text"].to_list()
        if len(texts):
            lengths = [len(t.split()) for t in texts]
            mean, std = np.mean(lengths), np.std(lengths)
            print("Mean sentence length:", mean, "Std:", std, end="\n\n")
       
    if fp is not None:
        new_df.to_csv(fp, index=False)
        print(fp + " created!")

    return new_df
