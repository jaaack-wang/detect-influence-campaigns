import re
import spacy
import torch 
import pandas as pd
from tqdm import tqdm
from time import time


def load_spacy(spacy_model):
    global nlp
    nlp = spacy.load(spacy_model, disable=["ner"])

    
def extract_parentheses_content(s):
    stack = []
    contents = []
    start = 0

    for i, char in enumerate(s):
        if char == '(':
            if not stack:
                start = i + 1
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
                if not stack:
                    contents.append(s[start:i])
    return contents



def tree2triplet(tree):
    triplets = []
    contents = extract_parentheses_content(tree)

    for values in contents:
        if 'nest' not in values:
            values = values.split(' ')
            triplets.append(
                ('AUTHOR', values[0], values[-1])
            )
        else:
            top_source = values[:values.find("nest") - 1]
            values = values[values.find("nest") + len("nest"):]
            values_nest = extract_parentheses_content(values)
            for nest_v in values_nest:
                if 'nest' not in nest_v:
                    nest_v = nest_v.split(' ')
                    triplets.append(
                        (f'AUTHOR_{top_source}', nest_v[0], nest_v[-1])
                    )
                else:
                    top_nest_source = nest_v[:nest_v.find("nest") - 1]
                    nest_v = nest_v[nest_v.find("nest") + len("nest"):]
                    values_nest_nest = extract_parentheses_content(nest_v)
                    for nest_v_nest in values_nest_nest:
                        nest_v_nest = nest_v_nest.split(' ')
                        triplets.append(
                            (f'AUTHOR_{top_source}_{top_nest_source}', nest_v_nest[0], nest_v_nest[-1])
                        )
    return triplets


def editDistDP(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
    
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
                
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                
            else:
                dp[i][j] = 1 + min(dp[i][j-1], # insert
                                   dp[i-1][j], # remove
                                   dp[i-1][j-1]) # replace
    # the min distance         
    return dp[m][n]


def repair_triplet(triplet, sent_tks):
    source, target, label = triplet
    
    if len(target.strip()) == 0 or label not in [
        "true", "ptrue", "unknown", "pfalse", "false"]:
        return tuple()
    
    target_lemma = nlp(target)[0].lemma_
    for tk in sent_tks:
        if tk.text == target:
            return triplet
        
        if tk.lemma_ == target_lemma:
            return (source, tk.text, label)

    for tk in [tk for tk in sent_tks if abs(len(tk) - len(target)) <= 1]:
        if editDistDP(tk.text, target) == 1:
            return (source, tk.text, label)
    for tk in [tk for tk in sent_tks if abs(len(tk) - len(target)) <= 2]:
        if editDistDP(tk.text, target) == 2:
            return (source, tk.text, label)
    
    return tuple()


def repair_triplets(triplets, sent_tks):
    out = []
    sent_tks = set(sent_tks)
    
    for triplet in triplets:
        triplet = repair_triplet(triplet, sent_tks)
        if len(triplet) != 0:
            out.append(triplet)

    return out


def remove_duplicates_and_get_target_ixes(triplets, sent_tks):
        
    out = []
    source_target_dict = dict()
    
    for triplet in triplets:
        prev = source_target_dict.get(triplet[:2], [])
        source_target_dict[triplet[:2]] = prev + [triplet]
    
    targets_ixes = []
    words_2_ix, lemma_2_ix = dict(), dict()
    for ix, tk in enumerate(sent_tks):
        word, lemma = tk.text, tk.lemma_
        words_2_ix[word] = words_2_ix.get(word, []) + [ix]
        lemma_2_ix[lemma] = lemma_2_ix.get(lemma, []) + [ix]
    
    for source_target, triplet_list in source_target_dict.items():
        count = len(triplet_list)
        target = source_target[1]
        target_lemma = nlp(target)[0].lemma_
        
        if target_lemma in lemma_2_ix:
            ixes = lemma_2_ix[target_lemma]
        elif target in words_2_ix:
            ixes = words_2_ix[target]
        else:
            continue
        
        if count == 1:
            out.append(triplet_list[0])
            targets_ixes.append(ixes[0])
        else:
            n = len(ixes)
            if count <= n:
                out.extend(triplet_list)
                targets_ixes.extend(ixes[:count])
            else:
                out.extend(triplet_list[:n])
                targets_ixes.extend(ixes[:n])
    
    return out, targets_ixes


def get_final_span(syntactic_head_token, head_token):
    # mention subtree vs children distinction in meeting!
    syntactic_head_subtree = list(syntactic_head_token.subtree)

    relevant_tokens = []

    for token in syntactic_head_subtree:
        if token.dep_ in ['cc', 'conj'] and token.i > head_token.i:
            break
        relevant_tokens.append(token)

    left_edge = relevant_tokens[0].i
    right_edge = relevant_tokens[-1].i + 1

    return left_edge, right_edge


def get_head_span(doc, ix):
    head_token = doc[ix]

    # when above target, eliminate CC or CONJ arcs
    # if on non-FB-target verb mid-traversal, DO take CC or CONJ arcs
    # if hit AUX, don't take CC or CONJ - don't worry for now
    if head_token.dep_ == 'ROOT':
        syntactic_head_token = head_token
    else:
        syntactic_head_token = None
        ancestors = list(head_token.ancestors)
        ancestors.insert(0, head_token)

        if len(ancestors) == 1:
            syntactic_head_token = ancestors[0]
        else:
            for token in ancestors:
                if token.pos_ in ['PRON', 'PROPN', 'NOUN']:
                    syntactic_head_token = token
                    break
                elif token.pos_ in ['VERB', 'AUX']:
                    syntactic_head_token = token
                    break

            if syntactic_head_token is None:
                for token in ancestors:
                    if token.pos_ == 'NUM':
                        syntactic_head_token = token
                        break
                    
                    # this else condition is added to handle the excecption mentioned above
                    # not sure if this is desirable
                    else:
                        syntactic_head_token = head_token

    # postprocessing for CC and CONJ -- exclude child arcs with CC or CONJ
    span_start, span_end = get_final_span(syntactic_head_token, head_token)

    return doc[span_start: span_end].text


def parse_tagged_beliefs(tagged, text):
        
    triplets = tree2triplet(tagged)
    
    doc = nlp(text)
    sent_tks = [tk for tk in doc]
    triplets = repair_triplets(triplets, sent_tks)

    if len(triplets) == 0:
        return []

    triplets, targets_ixes = remove_duplicates_and_get_target_ixes(triplets, sent_tks)

    out = []
    for ix, target_ix in enumerate(targets_ixes):
        source, target, label = triplets[ix]
        span = get_head_span(doc, target_ix)
        out.append([source, target, label, span, text])
    
    return out

    
def create_text_spans_datasets(df,
                               output_fp=None, 
                               show_processing_info=True, 
                               spacy_model="en_core_web_lg"):
    
    print(f"\n\n{'#'*20} Creating text spans {'#'*20}\n")
    
    if show_processing_info:
        start = time()
    
    load_spacy(spacy_model)
    
    # ============ data ============
    df.dropna(subset=["tagged"], inplace=True)
    
    # ============ parsing ============
    data = []
    rest_info = [c for c in df.columns if c not in ["text", "tagged"]]
    cols = ["source", "target", "belief", "span", "text"] + rest_info
        
    for i in tqdm(df.index):
        sub = df.loc[i]
        text, tagged = sub[["text", "tagged"]]
        rest = sub[rest_info].to_list()
        parsed = parse_tagged_beliefs(tagged, text)

        if parsed:
            data.extend([p + rest for p in parsed])
    
    text_spans_df = pd.DataFrame(data, columns=cols)
    new_cols = ["docID", "sentID"] + [c for c in cols if c not in ["docID", "sentID"]]
    text_spans_df = text_spans_df[new_cols]

    if show_processing_info:
        end = time()
        print("Processing time (sec):", end-start)
        print("Number of sentences with tagged beliefs parsed:", len(text_spans_df))  
        print("Number of text spans created:", len(text_spans_df))
        
    if output_fp:
        text_spans_df.to_csv(output_fp, index=False)
        print(output_fp + " created!\n\n")
    
    return text_spans_df
