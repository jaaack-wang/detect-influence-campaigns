import spacy
import argparse
import pandas as pd
from time import time
from datetime import datetime

from os import makedirs
from os.path import join, exists

import torch

from scripts.belief_tagging import tag_belief
from scripts.preprocessing import create_preprocessed_dataset
from scripts.make_text_span import create_text_spans_datasets

now = datetime.today().strftime("%Y-%m-%d %H-%M")

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--raw_data_fp", type=str, default="", help="Filepath for the input raw data. Must be in csv format and have a column called 'text'.")
parser.add_argument('--save_dir', type=str, default=join("results", now), help="Directory name in which the (intermediate) results of the pipeline are saved. Defaults to 'results/now' where now is current timestamp (to minutes).")
parser.add_argument("--to_resume", type=str, default=False, help="Whether to skip steps already made previously and continue what was left. Defaults to False.")
parser.add_argument("--spacy_model", type=str, default="en_core_web_lg", help="spacy_model to use. Defaults to 'en_core_web_lg'.")
parser.add_argument("--use_gpu", type=str, default=False, help="Whether to run spacy under GPU. If True and GPU is available, then run gpu spacy. Defaults to False.")
parser.add_argument("--break_sents", type=str, default=True, help="Whether to enforce sentence segmentation in the workflow. Defaults to True.")
parser.add_argument("--lower_bound", type=int, default=5, help="The minimum num of tokens for a text to be included in the preprocessing step. Defaults to 10.")
parser.add_argument("--upper_bound", type=int, default=100, help="The maximum num of tokens for a text to be included in the preprocessing step. Defaults to 50.")
parser.add_argument("--max_sent_num", type=int, default=None, help="The maximum num of sentences to keep if break_sents is set True. Defaults to None (unlimited).")
parser.add_argument("--show_processing_info", type=str, default=True, help="Whether to display useful processing info during the running of the workflow. Defaults to True.")
parser.add_argument("--cuda_device_num", type=str, default=0, help="Which cuda to use for deploying the belief tagging system.")
parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help="Directory name in which the checkpoints for the belief tagging system is saved. Defaults to checkpoints.")
parser.add_argument('--prefix', type=str, default="", help="Prefix for the finetuned belief-tagging system. Defaults to ''.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for the belief tagging system. Defaults to 10.")
parser.add_argument("--max_in_length", type=int, default=100, help="Maximum input token size for the belief tagging system.")
parser.add_argument("--max_out_length", type=int, default=200, help="Maximum output token size for the belief tagging system.")


def bool_eval(name, boolean):
    if isinstance(boolean, bool):
        return boolean
    
    boolean = boolean.lower()
    if boolean == "true":
        return True
    if boolean == "false":
        return False
    raise TypeError(f"{name} must be boolean, but {type(boolean)} was given.")


if __name__ == "__main__":
    args = parser.parse_args()
    makedirs(args.save_dir, exist_ok=True)
    
    args.use_gpu = bool_eval("use_gpu", args.use_gpu)
    args.to_resume = bool_eval("to_resume", args.to_resume)
    args.break_sents = bool_eval("break_sents", args.break_sents)
    args.max_in_length = max(args.upper_bound, args.max_in_length)
    args.show_processing_info = bool_eval("show_processing_info", args.show_processing_info)

    if args.to_resume:
        print("\nto_resume is set True (by default). That means, if an intermediate file from", end="")
        print(" a processing step already exists, that step will be skipped.", end="")
        print(" Set to_resume to False, if you want to be informed of an pre-existing processing step made earlier.\n")
    
    has_gpu = torch.cuda.is_available()
    if args.use_gpu and has_gpu:
        spacy.require_gpu()
    else:
        spacy.require_cpu()
        args.use_gpu = False
    
    filename = args.raw_data_fp.split("/")[-1].replace(".csv", "")
    
    data = pd.read_csv(args.raw_data_fp)
        
    def get_fp(suffix):
        if args.save_dir:
            return join(args.save_dir, f"{filename}_{suffix}.csv")
        return None

    def to_skip(fp):
        '''This function helps avoid repeating the same work'''
        skip = ""
        count = 0
        if exists(fp):
            skip = "y" if args.to_resume else ""
            
            while skip not in ["y", "n"]:
                if count > 0:
                    print("Please only enter 'y' (yes) or 'n' (no). Case insentive.\n")
                skip = input(fp + " already exists. Do you want to skip (y/n)\n?")
                skip = skip.lower()
                count += 1
        
        if skip == "y":
            df = pd.read_csv(fp)
            print(fp + " already exists and has been loaded\n")
            return df
        return None
    
    preprocessed = to_skip(get_fp("preprocessed"))
    if preprocessed is None:
        preprocessed = create_preprocessed_dataset(df=data, 
                                                   use_gpu=args.use_gpu,
                                                   fp=get_fp("preprocessed"), 
                                                   spacy_model=args.spacy_model,
                                                   break_sents=args.break_sents,
                                                   lower_bound=args.lower_bound, 
                                                   upper_bound=args.upper_bound,
                                                   max_sent_num=args.max_sent_num,
                                                   show_processing_info=args.show_processing_info)
    
    belief_tagged = to_skip(get_fp("belief_tagged"))
    if belief_tagged is None:
#     if belief_tagged is None or len(belief_tagged.dropna()) != len(preprocessed):
        # if only part of the preprocessed texts is belief tagged, then belief tag the rest 
        belief_tagged = tag_belief(inputs=preprocessed,
                                   ckpt_dir=args.ckpt_dir,
                                   prefix=args.prefix, 
                                   batch_size=args.batch_size,
                                   output_fp=get_fp("belief_tagged"), 
                                   max_in_length=args.max_in_length,
                                   max_out_length=args.max_out_length,
                                   cuda_device_num=args.cuda_device_num)
    
    text_spans_df = to_skip(get_fp("text_spans"))
    if text_spans_df is None:
        text_spans_df = create_text_spans_datasets(belief_tagged, 
                                                   spacy_model=args.spacy_model,
                                                   output_fp=get_fp("text_spans"), 
                                                   show_processing_info=args.show_processing_info)
