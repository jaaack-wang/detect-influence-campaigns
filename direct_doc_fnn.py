import argparse

import copy 
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join 
from scripts.utils import read_pkl_file

import torch
from torch import nn
from torch import optim 

from scripts.models import FNN
from scripts.train_eval_utils import *

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--save_dir', type=str, default="experiments/doc-level", help="Dire where the experiment data was saved.")
parser.add_argument("--cuda_device_num", type=str, default="0", help="Which cuda device to use.")
parser.add_argument("--run_num", type=int, default=5, help="Number of runs.")


def split_training_data(training_data, split_ratio=0.8):
    training_data = training_data.copy()
    np.random.shuffle(training_data)
    split = int(len(training_data) * split_ratio)
    train, val = training_data[:split], training_data[split:]
    return train, val


if __name__ == "__main__":
    
    args = parser.parse_args()
    save_dir = args.save_dir
    
    training_data = read_pkl_file(join(save_dir, "training_data.pkl"))
    test_data = read_pkl_file(join(save_dir, "test_data.pkl"))
    
    train_set, val_set = split_training_data(training_data)

    train_dl = create_dataloader(train_set, shuffle=False, batch_size=64)
    val_dl = create_dataloader(val_set, shuffle=False, batch_size=64)
    test_dl = create_dataloader(test_data, shuffle=False, batch_size=len(test_data))
    
    training_dl = create_dataloader(training_data, shuffle=False, batch_size=127)
    
    cuda_device_num = 0
    device = torch.device(f"cuda:{args.cuda_device_num}" if torch.cuda.is_available() else "cpu")
    print("Device availble in your current runtime:", device)
    
    run_num = args.run_num
    hid_dim = [90, 60, 30]
    num_epoch = 500
    learning_rate = 0.0005 
    print_freq = 5
    
    clf_dir = join(save_dir, "classification")
    makedirs(clf_dir, exist_ok=True)
    summary = []
    summary_cols = ["run", "dataset", "P", "R", "F1"]
    
    for run in range(1, run_num+1):
        
        print(f"\n{'#' * 20} Run#{run} {'#' * 20}\n")
        
        best_f1 = 0
        best_model = None
        
        model = FNN(len(train_set[0][0]), hid_dim, 2, torch.tanh).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(1, num_epoch+1):
            train_res =  train_loop(model, train_dl, optimizer, criterion, device)
            val_res = evaluate(model, val_dl, criterion, device)

            if epoch % print_freq == 0:
                print("Training Epoch: {}\nTrain: {}\nVal: {}\n".format(epoch, train_res, val_res))

            if val_res[-1] > best_f1:
                best_f1 = val_res[-1]
                best_model = copy.deepcopy(model)
        
        if best_model is not None:
            model = copy.deepcopy(best_model)
        
        torch.save(model.state_dict(), join(clf_dir, f"model_{run}.pt"))
        
        train_perf = evaluate(model, training_dl, criterion, device)
        test_perf = evaluate(model, test_dl, criterion, device)
        
        print("Training results:\nTrain: {}\nTest: {}\n".format(train_perf, test_perf))
        
        summary.append([run, "train"] + train_perf)
        summary.append([run, "test"] + test_perf)
    
    summary = pd.DataFrame(summary, columns=summary_cols)
    summary.to_csv(join(clf_dir, "summary.csv"), index=False)
    print("\n\n" + join(clf_dir, "summary.csv") + " has been saved!")
