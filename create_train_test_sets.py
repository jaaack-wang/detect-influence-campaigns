import argparse

import os
from os.path import join, exists

import json
import random
import pandas as pd
from scripts.utils import save_as_pkl, read_pkl_file

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--raw_data_fp', type=str, default="", required=True, help="Filepath to the raw data. Defaults to ''.")
parser.add_argument('--preprocessed_fp', type=str, default="", required=True, help="Filepath to the preprocessed data. Defaults to ''.")
parser.add_argument('--text_span_fp', type=str, default="", required=True, help="Filepath to the text span data. Defaults to ''.")
parser.add_argument('--test_set_ratio', type=float, default=0.2, help="Test set ratio. Defaults to 0.2.")
parser.add_argument('--save_dir', type=str, default="", required=True, help="Directory to save the results. Defaults to ''.")


def split_train_test_docIDs():
    text2docID = dict()
    test_docIDs = set()

    for ix in data.index:
        text = data.at[ix, "text"]
        text2docID[text] = text2docID.get(text, []) + [ix]

    lb = int(len(docIDs) * args.test_set_ratio)
    ub = int(lb * 1.05)

    while len(test_docIDs) < lb or len(test_docIDs) > ub:

        if len(test_docIDs) < lb:

            while True:
                docID = random.choice(docIDs)

                if docID not in test_docIDs:
                    new_docIDs = text2docID[data.at[docID, "text"]]
                    test_docIDs.update(set(new_docIDs))
                    break
        elif len(test_docIDs) > ub:

            while True:
                docID = random.choice(docIDs)

                if docID in test_docIDs:
                    existing_docIDs = text2docID[data.at[docID, "text"]]
                    test_docIDs -= existing_docIDs

    train_docIDs = set(docIDs) - test_docIDs
    print("# of docIDs:", len(docIDs))
    print("# of train docIDs:", len(train_docIDs))
    print("# of test docIDs:", len(test_docIDs))

    train_test_docIDs = {"train": list(train_docIDs), "test": list(test_docIDs)}
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_as_pkl(train_test_docIDs, join(args.save_dir, "train_test_docIDs.pkl"))
    return train_test_docIDs


def create_train_test_sets(df, train_test_docIDs, save_dir):
    train_docIDs = train_test_docIDs["train"]
    test_docIDs = train_test_docIDs["test"]
    
    if "docID" not in df:
        train_set = df.loc[train_docIDs]
        train_set.insert(0, "docID", train_docIDs)
        
        test_set = df.loc[test_docIDs]
        test_set.insert(0, "docID", test_docIDs)
        
        assert len(train_set) == len(train_docIDs)
        assert len(test_set) == len(test_docIDs)
        
        print("# of docs in the train set:", len(train_set))
        print("# of docs in the test set:", len(test_set))
        
    else:
        train_set = df[df.docID.isin(train_docIDs)]
        test_set = df[df.docID.isin(test_docIDs)]
        
        assert set(train_set.docID) - set(train_docIDs) == set()
        assert set(test_set.docID) - set(test_docIDs) == set()
        
        print("# of texts in the train set:", len(train_set))
        print("# of texts in the test set:", len(test_set))
        
        print("# of docs in the train set:", train_set.docID.unique().size)
        print("# of docs in the test set:", test_set.docID.unique().size)
    
    os.makedirs(save_dir, exist_ok=True)
    train_fp = join(save_dir, "train.csv")
    train_set.to_csv(train_fp, index=False)
    print(f"{train_fp} saved!")
    
    test_fp = join(save_dir, "test.csv")
    test_set.to_csv(test_fp, index=False)
    print(f"{test_fp} saved!")


if __name__ == "__main__":
    args = parser.parse_args()
    data = pd.read_csv(args.raw_data_fp)
    preprocessed = pd.read_csv(args.preprocessed_fp)
    text_spans = pd.read_csv(args.text_span_fp)
    
    docIDs = preprocessed.docID.unique().tolist()
    data = data.loc[docIDs]
    
    train_test_docIDs = split_train_test_docIDs()
    
    create_train_test_sets(data, train_test_docIDs, "experiments/doc-level")
    create_train_test_sets(preprocessed, train_test_docIDs, "experiments/sentence-level")
    create_train_test_sets(text_spans, train_test_docIDs, "experiments/target-level")
