import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import *

import pickle
from os import makedirs
from os.path import join, exists
import pandas as pd
import plotly.express as px


def get_clustering_summaries(clustering_dir, method=None):
    summary = pd.read_csv(join(clustering_dir, 
                               "clustering_summary.csv"))
    if method is not None:
        assert method in summary, f"No clustering results for {method}"
        summary = summary[summary.method == method]
    
    return summary

    
def get_a_single_clustering_result(res_dir, return_data=True, data_fp=None):
    
    cluster_table_fp = join(res_dir, "cluster_table.csv")
    class_2_text_indices_map_fp = join(res_dir, "class_2_text_indices_map.pkl")
    class_2_text_indices_map = read_pkl_file(class_2_text_indices_map_fp)  
    cluster_table = pd.read_csv(cluster_table_fp)
    
    if not return_data:
        return cluster_table, class_2_text_indices_map
    
    if data_fp is None:
        dirs = res_dir.split("/")
        assert "clustering" in dirs, "Please specify data_fp"
        ix = dirs.index("clustering")
        data_fp = "/".join(dirs[:ix+1]) + "/data.csv"
        
    data = pd.read_csv(data_fp)
    return data, cluster_table, class_2_text_indices_map


def get_cluster_res_dir_from_summary_table(clustering_dir, summary_table, row_ix):
    
    method, run, n, d, p, m, e = summary_table.loc[row_ix].to_list()[:7]
    cluster_fp = join(clustering_dir, f"{method}")
    
    res_fp = join(cluster_fp, f"dim={d} ")
    if method == "kmeans":
        res_fp += f"numOfClusters={n} run={run}"
    elif method == "hdbscan":
        res_fp += f"minPts={int(m)}"
    else:
        res_fp += f"minPts={int(m)} epsilon={e}"
    
    if method in ["hdbscan", "dbscan"]:
        res_fp += f" perplexity={int(p)} run={run}"
    
    if not exists(res_fp):
        res_fp = res_fp.replace(f"dim={d}", f"dim=full")
    
    return res_fp


def fill_data_with_cluster_labels(data, class_2_text_indices_map):
    data = data.copy()
    data.insert(0, "cluster", "")
    
    for c, ixes in class_2_text_indices_map.items():
        data.loc[ixes.tolist(), "cluster"] = c
    
    return data


def inspect_a_cluster(c, clustered_data, cluster_table, max_num_examples=5):
    cluster = clustered_data[clustered_data["cluster"] == c]
    
    if isinstance(max_num_examples, int):
        max_num_examples = min(max_num_examples, len(cluster))
        cluster = cluster.sample(max_num_examples)
    
    cluster_info = cluster_table[cluster_table.cluster == c].set_index("cluster").T
    return cluster, cluster_info


def visualize_a_clustering_result(res_dir, dim=2, reduce_method="umap", 
                                  perplexity=10, re_do_embd=False, 
                                  clustering_dir=None, data_fp=None, 
                                  width=None, height=None, show_plot=False,
                                  save_plot=False, save_fp=None):
    
    assert dim in [2, 3], "dim must be 2 or 3"
    
    dirs = res_dir.split("/")
    if clustering_dir is None:
        assert "clustering" in dirs, "Please specify clustering_dir"
        ix = dirs.index("clustering")
        clustering_dir = "/".join(dirs[:ix+1])
        
    if data_fp is None:
        data_fp = clustering_dir + "/data.csv"
        
    
    if "perplexity" in res_dir:
        perplexity = int(extract_value(r"(?<=perplexity=)\d+", res_dir))
    
    data = pd.read_csv(data_fp)
    
    embd_dir = join(clustering_dir, "embeddings")
    reduced_embd_fp = join(embd_dir, f"embds_dim={dim}.pkl")
    if exists(reduced_embd_fp) and not re_do_embd:
        embds = read_pkl_file(reduced_embd_fp)
    else:
        full_embd_fp = join(embd_dir, "embds_dim=full.pkl")
        assert exists(full_embd_fp), full_embd_fp + " does not exist!"
        embds = read_pkl_file(full_embd_fp)
        embds = get_dim_reduced_embds(embds, dim=dim, method=reduce_method, 
                                      perplexity=perplexity)
        
        save_as_pkl(embds, reduced_embd_fp)
    
    class_2_text_indices_map_fp = join(res_dir, "class_2_text_indices_map.pkl")
    class_2_text_indices_map = read_pkl_file(class_2_text_indices_map_fp)  

    
    data = fill_data_with_cluster_labels(data, class_2_text_indices_map)
    if dim==2:
        df = pd.DataFrame(embds).rename(columns={0:'x', 1:'y'})
        df = pd.concat([df, data], axis=1)
        fig = px.scatter(df, x='x', y='y',
                         color='cluster', hover_data=df.columns[2:],
                         width=width, height=height, labels={'color': 'label'},
                         title = f'2-d {reduce_method.upper()} visualization')

    else:
        df = pd.DataFrame(embds).rename(columns={0:'x', 1:'y', 2: "z"})
        df = pd.concat([df, data], axis=1)

        fig = px.scatter_3d(df, x='x', y='y', z="z",
                            color='cluster',
                            hover_data=df.columns[2:],
                            width=width, height=height,
                            labels={'color': 'label'},
                            opacity=0.7, size_max=18,
                            title = f'3-d {reduce_method.upper()} visualization')
    if show_plot:
        fig.show()

    if save_plot or save_fp is not None:
        if save_fp is None:
            save_fp = join(res_dir, reduce_method + f"-{dim}d")
        
        with open(f"{save_fp}.html", "w") as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))
            print(save_fp + " has been saved!")
