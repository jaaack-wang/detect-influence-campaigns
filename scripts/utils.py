'''
- Author: Zhengxiang (Jack) Wang 
- GitHub: https://github.com/jaaack-wang
- Website: https://jaaack-wang.eu.org
- About: Utility functions.
'''
import os
from os import listdir, walk
from os.path import join, isfile

import re
import pickle

from sentence_transformers import SentenceTransformer

import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_sbert_embedder(model_name_or_path):
    return SentenceTransformer(model_name_or_path)


def embed_texts_with_sbert(texts, sbert_embedder, batch_size=8):
    return sbert_embedder.encode(texts, batch_size= batch_size, 
                                 show_progress_bar=True)
    

def get_dim_reduced_embds(embds, dim=2, method="umap", perplexity=15):
    method = method.lower()
    assert method in [
                        "pca", "tsne", "umap"
                      ], "Supported reduction method: pca, tsne, umap."
    
    perplexity = min(perplexity, len(embds) // 2)
    
    if method == "pca":
        reducer = PCA(n_components=dim)
    elif method == "tsne":
        reducer = TSNE(n_components=dim, perplexity=perplexity)
    else:
        '''
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
        '''
        reducer = umap.UMAP(n_components=dim, n_neighbors=perplexity, 
                            metric='cosine', min_dist=0.0)

    X = reducer.fit_transform(embds)
    return X


def get_filepathes_from_dir(file_dir, include_sub_dir=False,
                            file_format=None, shuffle=False):
    
    if include_sub_dir:
        filepathes = []
        for root, _, files in walk(file_dir, topdown=False):
            for f in files:
                filepathes.append(join(root, f))
    else:
        filepathes = [join(file_dir, f) for f in listdir(file_dir)
                      if isfile(join(file_dir, f))]
        
    if file_format:
        if not isinstance(file_format, (str, list, tuple)):
            raise TypeError("file_format must be str, list or tuple.")
        file_format = tuple(file_format) if isinstance(file_format, list) else file_format
        format_checker = lambda f: f.endswith(file_format)
        filepathes = list(filter(format_checker, filepathes))

    if shuffle:
        random.shuffle(filepathes)
        
    return filepathes


def save_as_pkl(obj, fp, print_msg=True):
    with open(fp, "wb") as f:
        pickle.dump(obj, f)
        
    if print_msg:
        print(fp + " has been saved!")
        

def read_pkl_file(fp, print_msg=False):
    with open(fp, "rb") as f:
        out = pickle.load(f)
    
    if print_msg:
        print(fp + " has been loaded.\n")
    return out
    
    
def make_ngrams(tokens, n):
    out = []
    for i in range(0, len(tokens) - n + 1):
        out.append(' '.join(tokens[i:i+n]))
    return out


def tokenize(string, remove_stopwords=False):
    string = re.sub(r"[^-\w\s]", "", string)
    # stopwords from NLTK plus the following added items:
    # [would, -, also, could, couldn't]
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                 "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                 'same', 'so', 'than', 'too', 'very', 'can', 'cannot', 'will', 'just', "don't", 'should', "should've",
                 'now', "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                 "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't", "would", 
                 "-", "also", "could", "couldn't", "rt"]

    if remove_stopwords:
        return [tk for tk in string.lower().split() if tk not in stopwords]
    return string.lower().split()


def get_top_k_ngrams_and_avg_doc_freq_from_a_list(lst, n_gram=1, top_k=10):
    
    if not lst:
        return []

    ngrams = []
    doc_freqs = dict()

    remove_stopwords = True if n_gram == 1 else False
    
    for l in lst:
        tokens = tokenize(l, remove_stopwords)
        
        ngs = make_ngrams(tokens, n_gram)
        ngrams.extend(ngs)
        
        for ng in set(ngs):
            doc_freqs[ng] = doc_freqs.get(ng, 0) + 1
    
    top_k = min(top_k, len(doc_freqs))
    doc_freqs = sorted(doc_freqs.items(), 
                       key=lambda x: -x[1])[:top_k]
    
    top_k_ngram, avg_doc_freq = "", 0
    for i in range(len(doc_freqs)):
        top_k_ngram += (doc_freqs[i][0] + ", ")
        avg_doc_freq += doc_freqs[i][1]/len(lst)
    
    return top_k_ngram[:-2], avg_doc_freq / top_k


def get_top_k_ngrams_data(texts, top_k=10, ngram_range=(1, 3)):
    
    out = []
    ngram_sum = sum(range(ngram_range[0], ngram_range[1]+1))
    weighted_doc_freq = 0
    
    for n in range(ngram_range[0], ngram_range[1]+1):
        top_k_ngram, avg_doc_freq = get_top_k_ngrams_and_avg_doc_freq_from_a_list(
                                                        texts, n_gram=n, top_k=top_k)
        out.append(avg_doc_freq)
        out.insert(len(out)//2, top_k_ngram)
        weighted_doc_freq += (n/ngram_sum * avg_doc_freq)
            
    out.append(weighted_doc_freq)
    return out


def extract_value(pattern, text):
    out = re.search(pattern, text)
    if out:
        return out.group()
    return None
