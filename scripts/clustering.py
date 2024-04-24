import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import *

import numpy as np
import pandas as pd

from os import makedirs
from os.path import join, exists

from sentence_transformers.util import cos_sim

import hdbscan
from sklearn.cluster import DBSCAN, KMeans


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_class_2_text_indices_map(self, clusterers_classes):

    class_2_text_indices_map = dict()
    for c in np.unique(clusterers_classes):
        class_2_text_indices_map[f"C{c}"] = np.where(clusterers_classes==c)[0]

    return class_2_text_indices_map


def clustering(X, texts, method='HDBSCAN',
               n_clusters=10, min_cluster_size=10,  
               dist_metric='euclidean', epsilon=0.2):

    method = method.lower()
    if method=='dbscan':
        clusterer=DBSCAN(eps=epsilon, metric=dist_metric,
                         min_samples=min_cluster_size)
    elif method=='hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                    metric=dist_metric, prediction_data=True)
    elif method=='kmeans':
        clusterer=KMeans(n_clusters=n_clusters, n_init="auto")
        
    print(f"\n\n{'#'*20} Clustering {'#'*20}")
    clusterers_classes = clusterer.fit_predict(X)
    class_2_text_indices_map = get_class_2_text_indices_map(texts, 
                                                            clusterers_classes)
    print(f"\n\n{'#'*20} Clustering done {'#'*20}\n\n")
    return clusterer, class_2_text_indices_map


def make_cluster_table(class_2_text_indices_map, data, embds, text_col,
                       top_k=10, ngram_range=(1, 3), other_fields=[]):
    '''other_fields: valid fields in data. if given, their proportions inside
    a cluster will be calculated. '''
    out = []
    cols = ["cluster"]
    
    for n in range(ngram_range[0], ngram_range[1]+1):
        cols.append(f"top-{top_k} {n}-gram doc freq")
        cols.insert(len(cols)//2, f"top-{top_k} {n}-gram")
    
    cols = cols + ["weighted doc freq", "avg cos sim"]
    
    # sources and beliefs can be merged into other fields to make the code neater
    # but (1) for sources, we only want "% author_only", "% author_other", and
    # (2) for beliefs, we will ignore everything besides [true, false, unknown]
    
    if text_col == "span":
        cols += ["% author_only", "% author_other"]
        cols += ["% true", "% unknown", "% false"]
    
    other_fields_vals = []
    if other_fields:
        for field in other_fields:
            field_vals = data[field].unique().tolist()
            
            if len(field_vals) >= 10:
                print(f"There are {len(field_vals)} unique values in {field} column.")
                ans = input("You sure you want to move on? (type 'y' to continue): ")
                if ans.lower() != 'y':
                    raise RuntimeError("Clustering stops. Reconsider the other fields to include.") 
            
            other_fields_vals.append(field_vals)
            cols += ["% " + field_val for field_val in field_vals]
    
    cols += ["% unique docs", "cluster size"]
    
    texts = data[text_col]
    
    if text_col == "span":
        sources, beliefs = data["source"], data["belief"]
    
    for cls, txt_ixes in class_2_text_indices_map.items():
        
        # =================== texts =====================
        sub = [cls]
        cluster_texts = texts[txt_ixes].to_list()
        ngrams_data = get_top_k_ngrams_data(cluster_texts, top_k=top_k, 
                                            ngram_range=ngram_range)
        
        size = len(txt_ixes)
        cluster_embds = embds[txt_ixes.astype(np.int32)]
        cos_sim_matt = cos_sim(cluster_embds, cluster_embds)
        avg_cos_sim = (cos_sim_matt.sum() - size) / (size * size - size)
        
        sub.extend(ngrams_data + [avg_cos_sim.item()])
        
        if text_col == "span":
        # =================== sources =====================
            cluster_sources = sources[txt_ixes]
            percent_author_only = (cluster_sources == "AUTHOR").mean()
            sub.extend([percent_author_only, 1-percent_author_only])
        
        # ==================== belief ======================
            cluster_beliefs = beliefs[txt_ixes]
            percent_true = (cluster_beliefs == "true").mean()
            percent_unknow = (cluster_beliefs == "unknown").mean()
            percent_false = (cluster_beliefs == "false").mean()
            sub.extend([percent_true, percent_unknow, percent_false])
        
        # ============== other fields (if given) =============
        if other_fields:
            for field, field_vals in zip(other_fields, other_fields_vals):
                cluster_field = data[field][txt_ixes]
                for field_val in field_vals:
                    sub.append((cluster_field == field_val).mean())
        
        cluster_docID = data["docID"][txt_ixes]
        sub.append(cluster_docID.unique().size / len(txt_ixes))
        sub.append(size)
        out.append(sub)
    
    cluster_table = pd.DataFrame(out, columns=cols)
    cluster_table.sort_values(["avg cos sim"], ascending=False, ignore_index=True, inplace=True)
    return cluster_table


def make_clusters(save_dir, data=None, text_col="span", cluster_method='HDBSCAN', n_clusters=10, 
                  min_cluster_size=10, dist_metric='euclidean', epsilon=0.2,
                  sbert_model_name_or_path=None, lower_bound=10,
                  drop_duplicates=True, min_num_clusters=2,
                  reduced_dim=None, reduce_method="umap", perplexity=10, 
                  top_k=10, ngram_range=(1, 3), other_fields=[], batch_size=8,
                  re_do_data=False, re_do_embd=False, new_embed=False, make_returns=False):
    
    clustering_dir = join(save_dir, "clustering")
    makedirs(clustering_dir, exist_ok=True)
    
    # ================= checking input values =================
    cluster_method = cluster_method.lower()
    if cluster_method not in ["hdbscan", "dbscan", "kmeans"]:
        msg = f"Unsupported cluster method: {cluster_method}. "
        msg += "Supported methods: [HDBSCAN, DBSCAN, KMEANS]."
        raise RuntimeError(msg)
    else:
        cluster_fp = join(clustering_dir, cluster_method)
        makedirs(cluster_fp, exist_ok=True)
    
    def type_err_msg(arg, typ):
        return f"{arg} must be {typ}, when cluster_method='{cluster_method}'"
        
    if cluster_method == "kmeans":
        assert isinstance(n_clusters, int), type_err_msg("n_clusters", "int")
    elif cluster_method in ["hdsbcan", "dbscan"]:
        assert isinstance(min_cluster_size, int), type_err_msg("min_cluster_size", "int")
        if cluster_method == "dbscan":
            assert isinstance(epsilon, (float, int)), type_err_msg("epsilon", "float/int")   
    
    # ================= further checking if configured results exist =================
    if reduced_dim is not None:
        dim = reduced_dim
    else:
        dim = "full"
    
    res_fp = join(cluster_fp, f"dim={dim} ")
    if cluster_method == "hdbscan":
        res_fp += f"minPts={min_cluster_size}"
    elif cluster_method == "dbscan":
        res_fp += f"minPts={min_cluster_size} epsilon={epsilon}"
    elif cluster_method == "kmeans":
        res_fp += f"numOfClusters={n_clusters}"
    
    if reduced_dim is not None:
        res_fp += f" perplexity={perplexity}"
        
    run = 1
    while exists(res_fp + f" run={run}"):
        if cluster_method in ["hdbscan", "dbscan"] and (not new_embed) and (not re_do_embd):
            msg = f"{cluster_method} is deterministic with same text embeddings"
            msg += f" as well as same parameters: {res_fp}. please set re_do_embd"
            msg += " to True or change parameters if you intend to get different results."
            print(msg)
            break
        
        run += 1
    
    if run == 1:
        re_do_embd = False
        
    
    # ================= getting data =================
    def mark(text):
        if len(text.split()) < lower_bound:
            return False
        return True
    
    data_fp = join(clustering_dir, "data.csv")
    if not exists(data_fp) or re_do_data:
        msg = f"Please provide specify data (DataFrame) when {data_fp} "
        msg += "does not exist, or when re_do_data is set to True"
        assert data is not None, msg
        data = data.copy()
        data = data[data[text_col].apply(mark)]
        
        data.dropna(subset=[text_col], inplace=True)
        if drop_duplicates:
            cols_ = [c for c in data.columns if c != "sentID"]
            data.drop_duplicates(subset=cols_, inplace=True)
        
        data.reset_index(drop=True, inplace=True)
        data.to_csv(data_fp, index=False)
        print(data_fp + " has been saved.")
    else:
        data = pd.read_csv(data_fp)
    
    # ================= get embeddings =================
    embd_dir = join(clustering_dir, "embeddings")
    makedirs(embd_dir, exist_ok=True)
    full_embd_fp = join(embd_dir, "embds_dim=full.pkl")
        
    if exists(full_embd_fp):
        full_embds = read_pkl_file(full_embd_fp)
    else:
        sbert_embedder = get_sbert_embedder(sbert_model_name_or_path)
        full_embds = embed_texts_with_sbert(data[text_col], sbert_embedder, batch_size)
        save_as_pkl(full_embds, full_embd_fp)
        
    if reduced_dim is not None:
        reduced_embd_fp = join(embd_dir, f"embds_dim={reduced_dim}.pkl")
        if exists(reduced_embd_fp) and not re_do_embd:
            embds = read_pkl_file(reduced_embd_fp)
        else:
            embds = get_dim_reduced_embds(full_embds, reduced_dim, reduce_method, perplexity)
            save_as_pkl(embds, reduced_embd_fp)
    else:
        embds = full_embds
    
    # ================= get clustering results =================
    res_fp = res_fp + f" run={run}"
    class_2_text_indices_map_fp = join(res_fp, "class_2_text_indices_map.pkl")
    cluster_table_fp = join(res_fp, "cluster_table.csv")
    
    if exists(class_2_text_indices_map_fp):
        class_2_text_indices_map = read_pkl_file(class_2_text_indices_map_fp)
    else:
        clusterer, class_2_text_indices_map = clustering(embds, data['text'], 
                                                         cluster_method, 
                                                         n_clusters, 
                                                         min_cluster_size, 
                                                         dist_metric, epsilon)
        num_clusters = len(class_2_text_indices_map)
        if num_clusters < min_num_clusters:
            print(f"The number of clusters ({num_clusters}) < ", end="")
            print(f"min_num_clusters ({min_num_clusters}). Clusters are not saved.")
            return None, None, None

        makedirs(res_fp, exist_ok=True)
        save_as_pkl(class_2_text_indices_map, class_2_text_indices_map_fp)
        
    if exists(cluster_table_fp):
        cluster_table = pd.read_csv(cluster_table_fp)
        print(cluster_table_fp + " already exists and has been loaded.\n")
    else:
        cluster_table = make_cluster_table(class_2_text_indices_map, data, full_embds, 
                                           text_col, top_k, ngram_range, other_fields)
        cluster_table.to_csv(cluster_table_fp, index=False)
        print(join(res_fp, "cluster_table.csv") + " has been saved!\n")
    
    if make_returns:
        return cluster_table, class_2_text_indices_map, data
