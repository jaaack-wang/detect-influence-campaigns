
import LFE
import numpy as np

import pandas as pd
from os.path import join
from os import listdir
from tqdm import tqdm

from scripts.utils import *


texts ='''load your texts'''
feature_counts = []
lemmas = []

for text in tqdm(texts):
    f, l = LFE.get_text_feature_counts(text)
    feature_counts.append(f)
    lemmas.append(l)

feature_counts = np.array(feature_counts)
lemmas = np.array(lemmas)

assert feature_counts.shape[0] == len(texts)


def get_ling_features(ixes):
    counts = feature_counts[ixes]
    total_num_words = counts[:,0].sum()
    avg_fre_per_word = counts[:, 1:].sum(axis=0) / total_num_words
    types = set()

    for ls in lemmas[ixes]:
        types.update(ls)
            
    type_tk_ratio = len(types) / total_num_words
    
    return [type_tk_ratio] + avg_fre_per_word.tolist()
    

def add_ling_features_2_cluster_table(cluster_table, 
                                      class_2_text_indices_map, 
                                      fp=None, make_return=True):

    data = []

    for c in cluster_table.cluster:
        ixes = class_2_text_indices_map[c].astype(np.int32)
        data.append(get_ling_features(ixes))
        
    data = pd.DataFrame(data, columns=cols)
    cluster_table = pd.concat([cluster_table, data], axis=1)
            
    if fp:
        cluster_table.to_csv(fp, index=False)
        print(fp + " saved!")
    
    if fp is None:
        make_return = True
    
    if make_return:
        return cluster_table
