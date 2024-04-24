import argparse
import pandas as pd

import os
from os.path import join, exists
from scripts.clustering import make_clusters
from scripts.utils import get_filepathes_from_dir
from scripts.utils import read_pkl_file
from scripts.utils import extract_value
from scripts.explore_clustering_results import visualize_a_clustering_result


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--save_dir', type=str, default="results/test", help="Directory name in which the (intermediate) results of the pipeline are saved. Defaults to 'results/test'.")
parser.add_argument('--data_fp', type=str, default="", help="Filepath to the data_fp. Defaults to ''.")
parser.add_argument('--text_col', type=str, default="text", help="Name of the text column to be clustered.")
parser.add_argument("--drop_duplicates", type=str, default=False, help="Whether to drop duplicates in data. Defaults to False.")
parser.add_argument("--text_lower_bound", type=int, default=5, help="Min length for the text to be clustered. Defaults to 5.")
parser.add_argument('--sbert_model_name_or_path', type=str, default="all-mpnet-base-v2", help="Name or path to a sbert model. Can be None if you already run a clustering experiment once. Defaults to 'all-mpnet-base-v'.")
parser.add_argument("--sbert_batch_size", type=int, default=8, help="Batch size for sbert encoding. Defaults to 8.")
parser.add_argument("--cuda_device_num", type=str, default="0", help="Which cuda device to use for sbert.")
parser.add_argument("--num_of_runs", type=int, default=1, help="Num of runs for the clustering experiment(s). Defaults to 1.")
parser.add_argument("--cluster_method", type=str, default="kmeans", help="Clustering method(s): [kmeans, hdbscan, dbscan]. If more than one, use commas to separate them. Defaults to [kmeans].")
parser.add_argument("--n_clusters", type=str, default="10", help="Number of clusters for kmeans. If more than one, use commas to separate them. Defaults to [10].")
parser.add_argument("--min_cluster_size", type=str, default="5", help="Min cluster size for hdbscan or dbscan. If more than one, use commas to separate them. Defaults to [5].")
parser.add_argument("--dist_metric", type=str, default="euclidean", help="Distance metric for hdbscan or dbscan. Defaults to 'euclidean'. More see: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#distance-matrices")
parser.add_argument("--min_num_clusters", type=int, default=2, help="Min num clusters to save a clustering results. Defaults to 2.")
parser.add_argument("--reduced_dim", type=str, default="10", help="The dim you want the sbert embeddings to be reduced to, particularly necessarily for hdbscan or dbscan. If more than one, use commas to separate them. Defaults to [10].")
parser.add_argument('--reduce_method', type=str, default="umap", help="Dimensionality reduction algoirthm to use. Defaults to umap. Also supports tsne(2/3 dim only) and PCA.")
parser.add_argument("--reduced_dim_for_kmeans", type=str, default=False, help="Whether to use reduced sbert embeddings for kmeans clustering. Defaults to False.")
parser.add_argument("--perplexity", type=str, default="10", help="The size of local neighborhood used for manifold approximation, specific to umap and tsne. If more than one, use commas to separate them. Defaults to [10].")
parser.add_argument("--epsilon", type=str, default="0.2", help="Min distance for any pair in a non-outlier cluster, specific to dbscan. If more than one, use commas to separate them. Defaults to [0.2].")
parser.add_argument("--top_k", type=int, default=10, help="The top k n-grams to extract for each resulting cluster to make a cluster table. Defaults to 10.")
parser.add_argument("--ngram_range", type=str, default="1,3", help="The range of n-gram to extract and add into the cluster table. Must be a pair of numbers separated by commas. Defaults to [1, 3].")
parser.add_argument("--other_fields", type=str, default=[], help="Other filed(s) in your data you would like to inclde in the cluster table (proportion info). If more than one, use commas to separate them. Defaults to [].")
parser.add_argument("--re_do_embd", type=str, default=False, help="Whether to re-do embedding reduction, since the reduction algoirthms may not be deterministic. Defaults to False.")
parser.add_argument("--visualize_result", type=str, default=True, help="Whether to visualize the clustering results. Defaults to True.")
parser.add_argument("--visualizing_dim", type=str, default="2", help="Visualization dim. If more than one, use commas to separate them. Defaults to '2'.")


def list_to_type(string, typ_func=int):
        return list(map(typ_func, string.split(",")))
    
    
def bool_eval(name, boolean):
    if isinstance(boolean, bool):
        return boolean
    
    boolean = boolean.lower()
    if boolean == "true":
        return True
    if boolean == "false":
        return False
    raise TypeError(f"{name} must be boolean, but {type(boolean)} was given.")
    

def this_type_or_None(v, typ_func=int):
    return typ_func(v) if v else v


if __name__ == "__main__":
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_num
    
    # ========================= clustering =========================
    
    if not exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"{args.save_dir} does not exist, so it has been created!")
    
    if exists(args.data_fp):
        data = pd.read_csv(args.data_fp)
    else:
        fp = join(args.save_dir, "data.csv")
        assert exists(fp), f"{fp} does not exist, please specify data_fp"
    
    ngram_range = list_to_type(args.ngram_range)
    re_do_embd = bool_eval("re_do_embd", args.re_do_embd)
    reduced_dim_for_kmeans = bool_eval("reduced_dim_for_kmeans", args.reduced_dim_for_kmeans)
    drop_duplicates = bool_eval("drop_duplicates_and_na", args.drop_duplicates)
    visualize_result = bool_eval("visualize_result", args.visualize_result)
    if visualize_result:
        visualizing_dims = list_to_type(args.visualizing_dim)
    
    cluster_methods = [m.strip() for m in args.cluster_method.split(",")]
    if args.other_fields:
        args.other_fields = [o.strip() for o in args.other_fields.split(",")]
         
    for cluster_method in cluster_methods:   
        
        n_clusters = list_to_type(args.n_clusters)
        min_cluster_sizes = list_to_type(args.min_cluster_size)
        reduced_dims = list_to_type(args.reduced_dim)
        epsilons = list_to_type(args.epsilon, float)
        perplexities = list_to_type(args.perplexity)
    
        if cluster_method == "kmeans":
            if not reduced_dim_for_kmeans:
                reduced_dims = [None] 
            
            epsilons = [None]
            re_do_embd = False
            perplexities = [None]
            min_cluster_sizes = [None]
            min_cluster_sizes = [None]

        elif cluster_method in ["hdbscan", "dbscan"]:
            n_clusters = [None]
            re_do_embd = True

            if cluster_method == "hdbscan":
                epsilons = [None]

        print(f"\nUsing {cluster_method} clustering...\n")
        
        for _ in range(args.num_of_runs):
            for n in n_clusters:
                for d in reduced_dims:
                    new_embed = False
                    re_do_embd = args.re_do_embd
                    for p in perplexities:
                        for m in min_cluster_sizes:
                            for e in epsilons:
                                make_clusters(save_dir=args.save_dir, data=data, text_col=args.text_col,
                                              cluster_method=cluster_method, n_clusters=n, 
                                              min_cluster_size=m, dist_metric=args.dist_metric,
                                              epsilon=e, sbert_model_name_or_path=args.sbert_model_name_or_path, 
                                              lower_bound=args.text_lower_bound, 
                                              min_num_clusters=args.min_num_clusters,
                                              drop_duplicates=drop_duplicates,
                                              reduced_dim=d, reduce_method=args.reduce_method, perplexity=p,
                                              top_k=args.top_k, ngram_range=ngram_range, 
                                              other_fields=args.other_fields, batch_size=args.sbert_batch_size,
                                              re_do_data=False, re_do_embd=re_do_embd, new_embed=new_embed, make_returns=False)
                                re_do_embd = False
                                new_embed = True
                                              
    # ========================= summarizing clustering =========================
    clustering_dir = join(args.save_dir, f"clustering")
    cluster_table_fps = get_filepathes_from_dir(clustering_dir, 
                                                include_sub_dir=True, 
                                                file_format="cluster_table.csv")
    full_embd_fp = join(clustering_dir, "embeddings/embds_dim=full.pkl")
    full_dim = read_pkl_file(full_embd_fp, print_msg=False).shape[1]

    summaries = []
    cluster_table = pd.read_csv(cluster_table_fps[0])
    cols = cluster_table.columns.to_list()
    ix = cols.index("top-10 1-gram doc freq")
    cols = cols[ix:] 
    methods = set()

    for ix, fp in enumerate(cluster_table_fps):
        
        if visualize_result:
            for dim in visualizing_dims:
                res_dir = "/".join(fp.split("/")[:-1])
                visualize_a_clustering_result(res_dir, dim=dim, show_plot=False, save_plot=True)

        if ix > 0:
            cluster_table = pd.read_csv(fp)

        dirs = fp.split("/")
        method = dirs[-3]
        methods.add(method)

        parameters = dirs[-2]
        run = int(extract_value(r"(?<=run=)\d+", parameters))
        num_clusters = len(cluster_table)
        dim = extract_value(r"(?<=dim=)\S+", parameters)
        dim = full_dim if dim == "full" else int(dim)

        perplexity = extract_value(r"(?<=perplexity=)\d+", parameters)
        perplexity = this_type_or_None(perplexity)

        min_cluster_size = extract_value(r"(?<=minPts=)\d+", parameters)
        min_cluster_size = this_type_or_None(min_cluster_size)

        epsilon = extract_value(r"(?<=epsilon=)\S+", parameters)
        epsilon = this_type_or_None(epsilon, float)

        others = [method, run, num_clusters, dim, perplexity, 
                  min_cluster_size, epsilon]
        means = cluster_table[cols].mean().to_list()
        summaries.append(others + means)


    cols =  ["mean " + c for c in cols]
    cols = ["method", "run", "# clusters", "dim", 
            "perplexity", "min cluster size", "epsilon"] + cols


    summary = pd.DataFrame(summaries, columns=cols)
    fp = join(clustering_dir, "clustering_summary.csv")
    summary.to_csv(fp, index=False)
    print(fp + " has been saved!")
