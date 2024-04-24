import pandas as pd

import json
from tqdm import tqdm
from os.path import join, exists

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ToDataset(Dataset):
    def __init__(self, tokenizer, inputs, max_len=50):

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []

        self._build_examples(inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask}

    def _build_examples(self, inputs):

        for inpt in inputs:
            
            tokenized_input = self.tokenizer.batch_encode_plus(
                [inpt],
                max_length=self.max_len,
                pad_to_max_length=True,
                truncation=True,
                return_tensors="pt",)

            self.inputs.append(tokenized_input)


def generate(data_loader, model, tokenizer, device, max_out_length):
    model.eval()
    outputs = []
    for batch in tqdm(data_loader):
        outs = model.generate(input_ids=batch['source_ids'].to(device),
                              attention_mask=batch['source_mask'].to(device),
                              max_length=max_out_length)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        outputs.extend(dec)

    return outputs


def tag_belief(inputs, ckpt_dir, prefix="",
               output_fp=None, batch_size=100, 
               max_in_length=50, max_out_length=100, cuda_device_num=0):

    print(f"\n\n{'#'*20} Belief tagging {'#'*20}")
    
    config = json.load(open(join(ckpt_dir, "config.json"), "r"))
    tokenizer = T5Tokenizer.from_pretrained(config["_name_or_path"])
    device = torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() else 'cpu')

    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
    model.load_state_dict(torch.load(join(ckpt_dir, "pytorch_model.bin"), map_location="cpu"))
    model.to(device)
        
    texts = inputs["text"].unique().tolist()
    if prefix:
        texts_ = [prefix+t for t in texts]
    else:
        texts_ = texts
    
    ds = ToDataset(tokenizer, texts_, max_len=max_in_length)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)
    tagged = generate(dl, model, tokenizer, device, max_out_length)
    
    outputs = inputs["text"].map(dict(zip(texts, tagged)))
    inputs.insert(2, "tagged", outputs)
    
    if output_fp:
        inputs.to_csv(output_fp, index=False)
        print(output_fp + " created!\n\n")
     
    return inputs
