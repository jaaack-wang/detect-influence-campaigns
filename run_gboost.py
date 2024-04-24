import argparse
import pandas as pd

from os import makedirs
from os.path import join 
from collections import Counter

import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
from scripts.utils import read_pkl_file
from sklearn.metrics import precision_recall_fscore_support


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--save_dir', type=str, default="results/test", help="Directory name in which the (intermediate) results of the pipeline are saved. Defaults to 'results/test'.")
parser.add_argument("--run_num", type=int, default=5, help="Number of runs.")
parser.add_argument("--model", type=str, default="xgboost", help="Gboost model to use. If not xgboost, use sklearn GradientBoostingClassifier.")

labels = ["% pos", "% neg"]
num_features = ['top-10 1-gram doc freq', 'top-10 2-gram doc freq', 'top-10 3-gram doc freq', 
                'weighted doc freq', 'avg cos sim', "% unique docs", "cluster size"]
ling_features = ['Type-token ratio', 'Mean word length', 'Six letter words and longer', 'Contraction', 
                 'Agentless passive', 'By passive', 'Past tense', 'Perfect aspect', 'Non-past tense', 
                 'Progressive tense', 'Split auxiliaries', 'Phrasal Coord', 'Independent clause coord', 
                 'WH question', 'WH clause', 'That relative', 'WH relative on subject position', 
                 'WH relative on object position', 'WH relative with fronted prep', 'Past participial clause', 
                 'Speech act verb + to', 'Cognition verb + to', 'Desire verb + to', 'Modality verb + to', 
                 'Probability verb + to', 'Certainty adj + to', 'Ability adj + to', 'Personal affect adj + to', 
                 'Ease_difficulty adj + to', 'Evaluative adj + to', 'Control noun + to', 
                 'Nonfactive noun + that', 'Attitudinal noun + that', 'Factive noun + that', 
                 'Likelihood noun + that', 'Nonfactive verb + that', 'Attitudinal verb + that', 
                 'Factive verb + that', 'Likelihood verb + that', 'Likelihood adj + that', 
                 'Attitudinal adj + that', 'Noun', 'VERB', 'Noun modifier', 'Article', 'Modal', 
                 'Negator', 'Preposition', 'First person pronoun singular', 'First person pronoun plural', 
                 'Second person pronoun', 'Third person pronoun', 'Pronoun it', 'Demonstrative pronoun', 
                 'Indefinite pronoun', 'Nominalization', 'Animate noun', 'Cognitive noun', 'Concrete noun', 
                 'Technical noun', 'Quantity noun', 'Place noun', 'Group noun', 'Abstract noun', 
                 'Be as main verb', 'Pro-verb do', 'Activity verb', 'Communication verb', 'Mental verb', 
                 'Causative verb', 'Ocurrence verb', 'Existence verb', 'Aspectual verb', 'Attributive adj', 
                 'Predictive adjective', 'Place adverb', 'Time adverb', 'Nonfactive adverb', 
                 'Attitudinal adverb', 'Factive adverb', 'Likelihood adverb', 'Causative subordinator', 
                 'Conditional subordinator', 'Contrastive subordinator', 'Other subordinator', 
                 'Possibility modal', 'Necessity modal', 'Predictive modal', 'Conjunct', 'Downtoner', 
                 'Amplifier', 'Hedge', 'Emphatics', 'polite expression', 'Evidential expression']

# features = num_features
# features = ling_features

features = num_features + ling_features


def make_labels(cluster_table, alpha=0.95):
    labels = []
    
    for ix in cluster_table.index:
        percent_pos = cluster_table.at[ix, "% pos"]        
        if percent_pos >= alpha:
            labels.append(1)
        else:
            labels.append(0)
        
    return labels


def make_datasets(cluster_table):
    
    X = cluster_table[features].to_numpy()
    Y = make_labels(cluster_table)
    return X, Y


def evaluate(predict, X, Y):
    Ypred = predict(X)
    *scores, _ = precision_recall_fscore_support(Y, Ypred, 
                                                 average="binary", 
                                                 zero_division=0)
    return scores


def fill_data_with_cluster_labels(data, class_2_text_indices_map, return_full_table=False):
    data = data.copy()
    
    if not return_full_table:
        data = data[["docID", "label"]]
    
    data.insert(0, "cluster", "")
    
    for c, ixes in class_2_text_indices_map.items():
        data.loc[ixes.tolist(), "cluster"] = c
    
    return data


def predict_cluster_table(model, cluster_table):
    X, Y = make_datasets(cluster_table)
    predictions = model.predict(X)
    confidence = model.predict_proba(X)[:, 1]
    *scores, _ = precision_recall_fscore_support(Y, predictions, 
                                                 average="binary", 
                                                 zero_division=0)
    return predictions, scores, confidence


def evaluate_hdoi_clusters(hdoi_clusters):
    influenced_docIDs = set()
    for c in hdoi_clusters.cluster:
        docIDs = clustered_data.docID[class_2_text_indices_map[c]]
        influenced_docIDs.update(set(docIDs))

    preds = gold_label_table.docID.isin(influenced_docIDs)
    scores = precision_recall_fscore_support(labels, preds, 
                                             average="binary", zero_division=0)
    return list(scores[:3])        
        
    
if __name__ == "__main__":
    args = parser.parse_args()
    save_dir = args.save_dir
    
    training_data = pd.read_csv(join(save_dir, "training_data.csv"))
    test_data = pd.read_csv(join(save_dir, "test_data.csv"))
    
    training_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    training_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    
    train_X, train_Y = make_datasets(training_data)
    test_X, test_Y = make_datasets(test_data)
    
    clf_dir = join(save_dir, "classification_gboost")
    makedirs(clf_dir, exist_ok=True)
    
    summary = []
    summary_cols = ["run", "dataset", "type", "P", "R", "F1"]
    
    run_num = args.run_num
    for run in range(1, run_num+1):
        print(f"\n{'#' * 20} Run#{run} {'#' * 20}\n")
        
        res_dir = join(clf_dir, str(run))
        makedirs(res_dir, exist_ok=True)
        
        if args.model == "xgboost":
            model = xgb.XGBClassifier(max_depth=run).fit(train_X, train_Y)
        else:
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                   max_depth=1, random_state=run).fit(train_X, train_Y)
        
        for ds in ["train", "test"]:
            print(f"\n{'#'*20} Deploying to {ds} set {'#'*20}\n")

            if ds == "test":
                eval_data = test_data
            else:
                eval_data = training_data
                
            # ======================== individuals ========================

            fps = eval_data.filepath.unique() 
            
            clustering_dir = join(save_dir, f"results_{ds}/clustering")
            data = pd.read_csv(join(clustering_dir, "data.csv"))
            gold_label_table = pd.read_csv(f"experiments/doc-level/{ds}.csv")
            labels = (gold_label_table.label == "pos")

            results = []
            cols = ["fp", "model_precision_on_hdoi", "model_recall_on_hdoi", "model_f1_on_hdoi", 
                    "precision", "recall", "f1", "# hdoi", "avg % pos", "avg % unique docs", 
                    "avg confidence", "cluster size"]

            for fp in fps:
                cluster_table = eval_data.copy()[eval_data.filepath == fp]
                predictions, scores_on_hdoi, confidence = predict_cluster_table(model, cluster_table)

                if sum(predictions) == 0:
                    continue

                avg_confidence = sum([c for i,c in enumerate(confidence) if predictions[i]==1]) / sum(predictions)
                fp_ = join(clustering_dir, fp).replace("cluster_table.csv", "class_2_text_indices_map.pkl")
                class_2_text_indices_map = read_pkl_file(fp_)
                clustered_data = fill_data_with_cluster_labels(data, class_2_text_indices_map)

                cluster_table["predictions"] = predictions
                hdoi_clusters = cluster_table[cluster_table.predictions == 1]
                scores_on_docs = evaluate_hdoi_clusters(hdoi_clusters)

                results.append([fp] + scores_on_hdoi + scores_on_docs + [len(hdoi_clusters), 
                                                                         hdoi_clusters["% pos"].mean(),
                                                                         hdoi_clusters["% unique docs"].mean(),
                                                                         avg_confidence,
                                                                         len(cluster_table)])

            results_df = pd.DataFrame(results, columns=cols)
            results_df.to_csv(join(res_dir, f"individuals_{ds}.csv"), index=False)
            print(join(res_dir, f"individuals_{ds}.csv") + " has been saved!")

            # ======================== aggregation ========================
            id_counts = []
            total_hdoi = 0

            for fp in fps:
                cluster_table = eval_data.copy()[eval_data.filepath == fp]
                predictions, scores_on_hdoi, _ = predict_cluster_table(model, cluster_table)

                fp_ = join(clustering_dir, fp).replace("cluster_table.csv", "class_2_text_indices_map.pkl")
                class_2_text_indices_map = read_pkl_file(fp_)
                clustered_data = fill_data_with_cluster_labels(data, class_2_text_indices_map)

                cluster_table["predictions"] = predictions
                hdoi_clusters = cluster_table[cluster_table.predictions == 1]
                total_hdoi += len(hdoi_clusters)

                for c in hdoi_clusters.cluster:
                    docIDs = clustered_data.docID[class_2_text_indices_map[c]]
                    id_counts.extend(docIDs)

            out = []
            id_counter = Counter(id_counts).most_common()
            cols = ["% of total hdoi", "min freq", "precision", "recall", "f1"]

            for percent in range(0, 51, 1):
                percent = percent / 100
                min_freq = percent * total_hdoi
                ids = [i for i, f in id_counter if f >= min_freq]
                preds = gold_label_table.docID.isin(ids)
                scores = precision_recall_fscore_support(labels, preds, 
                                                         average="binary", zero_division=0)
                out.append([percent, min_freq] + list(scores[:3]))
            
            out = pd.DataFrame(out, columns=cols)
            out.to_csv(join(res_dir, f"aggregation_{ds}.csv"), index=False)
            print(join(res_dir, f"aggregation_{ds}.csv") + " has been saved!")
            
            # ======================== summaries ========================
            means = [run, ds, "mean"]
            medians = [run, ds, "median"]
            mmax = [run, ds, "max"]
            aggregates_5 = [run, ds, "5%"]
            aggregates_10 = [run, ds, "10%"]
            aggregates_15 = [run, ds, "15%"]
            aggregates_20 = [run, ds, "20%"]
            aggregates_25 = [run, ds, "25%"]
            aggregates_best = [run, ds, "best"]
            
            for m in ["precision", "recall", "f1"]:
                means.append(results_df[m].mean())
                medians.append(results_df[m].median())
                mmax.append(results_df[m].max())
                
                aggregates_5.append(out[out["% of total hdoi"] == 0.05][m].item())
                aggregates_10.append(out[out["% of total hdoi"] == 0.1][m].item())
                aggregates_15.append(out[out["% of total hdoi"] == 0.15][m].item())
                aggregates_20.append(out[out["% of total hdoi"] == 0.2][m].item())
                aggregates_25.append(out[out["% of total hdoi"] == 0.25][m].item())
                aggregates_best.append(out[out["f1"] == out["f1"].max()].iloc[0][m])
                
            summary.extend([means, medians, mmax, aggregates_5, aggregates_10, aggregates_15, 
                            aggregates_20, aggregates_25, aggregates_best])
    
    summary = pd.DataFrame(summary, columns=summary_cols)
    summary.to_csv(join(clf_dir, "summary.csv"), index=False)
    print("\n\n" + join(clf_dir, "summary.csv") + " has been saved!")
