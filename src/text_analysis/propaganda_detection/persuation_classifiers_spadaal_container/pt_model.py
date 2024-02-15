from roberta_multispan import *
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification, IntervalStrategy
from transformers.utils import logging
import torch 
import pandas as pd
import transformers
import numpy as np
import pickle 
import configparser
import pt_constants


label_dict = dict(pickle.load(open('label_dict-golden.pkl','rb')))
os.environ["WANDB_DISABLED"] = "true"
logging.set_verbosity_error() #disable warning

from more_itertools import chunked, pairwise, padded, windowed, flatten ,unzip


import argparse 
import sys

from torch.nn import Sigmoid
sigmoid = Sigmoid()

def read_config(filename='defaults,cfg'):
    '''
    Accetta in input un file cfg da cui ricavare 
    i parametri utili al modello e creare un parser. 
    '''
    config = configparser.ConfigParser()
    config.read('defaults.cfg')
    
    return config


def pool_results(predictions, indexes, cls_, pad, labels=pt_constants.label_dict):
    
    predictions = [w[1:-1] for w in predictions] #get rid of CLS + SEP tokens
    ind_arr = np.array(indexes) 

    windows = []
    
    for i in set(indexes): #groupby document
        for pos in np.where(ind_arr == i):
            windows.append([predictions[p] for p in pos])

        def pool_single(prediction_list):
            #pool
            
            if len(prediction_list) == 1:
                ret = prediction_list[0]
            else:
                ret = prediction_list[0][:255] # keep first half chunk
                ret.extend(flatten([np.maximum(first[255:],second[:255]).tolist() for first, second in pairwise(prediction_list)])) 
                ret.extend(flatten([prediction_list[-1][255:]])) # keep last half chunk

            return ret
    
    return list(map(pool_single, windows))

def make_chunks(docs, cls_, sep, pad):
    
    windows = map(lambda y: windowed(y, n=510, step=255, fillvalue=pad), docs)
    windows = map(lambda y: [[cls_] + list(w) + [sep]  for w in y], windows)
    return list(windows)


def preprocess_files(text, spans, label_dict):
    
    #text = pd.read_csv(text_path).set_index('id')
    #spans = pd.read_csv(spans_path)
    text = text.set_index('id')

    #Setup label encoder and decoders (simple dict)
    label_dict = dict(pickle.load(open(label_dict,'rb')))
    label_rev_dict = {v:k for k,v in label_dict.items()}
    
    #Make SPANs dataframe
    ranges = spans.groupby(['id','class']).agg(lambda x: list(x))
    ranges['span'] = ranges.apply(lambda row: list(zip(row['start'],row['end'])), axis=1)
    ranges= ranges.reset_index()
    ranges['labels_i'] = ranges['class'].replace(label_rev_dict)
    
    #Make the lists of text w/ spans
    to_tokenize = []
    to_make_mask = []
    indexes = []
    for article, view in ranges.groupby('id'):
        to_tokenize.append(text.loc[article]['text'])
        to_make_mask.append(dict(view[['labels_i','span']].values))
        indexes.append(article)
    #include documents with zero annotations ignored by groupby
    ind_ = text[~text.index.isin(spans.id)].index.values
    article_txt = text[~text.index.isin(spans.id)]['text'].values
    
    for ind, article_txt in zip(ind_,article_txt):
        to_tokenize.append(article_txt)
        to_make_mask.append({})
        indexes.append(ind)
        
    return (to_tokenize, to_make_mask, indexes)

    
def instantiate_model(cfg: dict):
    
    model_fp = cfg['model']['model_fp']
    batch_size = cfg['model']['batch_size']
    default_threshold = float(cfg['postprocessing']['threshold'])
    
    # download the pretrained model given name or path
    model = AutoModelForTokenClassification.from_pretrained(model_fp)
    model.eval()

    print(torch.cuda.empty_cache())
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print(device)
    model.to(device)
    
    batch_size = int(batch_size)
    
    args = TrainingArguments(model_fp,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy = IntervalStrategy.STEPS,
    eval_steps = 100,
    max_steps=4000,
    load_best_model_at_end = True,
    metric_for_best_model = 'f1',
    save_total_limit = 5
    )

    trainer = MultiLabelTrainer(
    model,
    args,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_fp)  # Could change the tokenizer to point to another path
    
    return {'trainer': trainer, 'tokenizer': tokenizer , 'default_threshold': default_threshold}

def tokenize_data_multispan(list_text, tokenizer):
    #test_txt, test_spans, indexes_test = preprocess_files(df_test_txt, df_test_spans, './label_dict-golden.pkl')
    tok_batch_test_full = tokenizer(list_text,padding=False, truncation=False, add_special_tokens=False)
    tok_batch_test = tok_batch_test_full.copy()
    offsets_test= list(map(lambda x: x.offsets, tok_batch_test_full.encodings))
    #test_y = list(map(calculate_y, zip(offsets_test, test_spans)))
    #test_y = [y + [([0] * 23)] for y in test_y] #add label for SEP token
    #dev_y=[[[tok[11],tok[12]] for tok in y] for y in dev_y] #only LL
    #tok_batch_dev['labels'] = dev_y
    ## IndexIDS
    cls_ = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id
    special_tokens = {'sep':sep, 'cls':cls_, 'pad': pad}
    
    
    input_ids_test = [ids + [sep] for ids in tok_batch_test_full['input_ids']]

    #Chunk TEst
    

    windows = make_chunks(input_ids_test, cls_, sep, pad)
    indexes_test = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    index_ids_chunked_test = windows_flat


    ## Att masks
    att_mask_chunked_test = [[pad] * len(chu) for chu in index_ids_chunked_test]

    ## put back on the Tokenizer Batch
    tok_batch_test['input_ids'] = index_ids_chunked_test
    tok_batch_test['attention_mask'] = att_mask_chunked_test


    #Final objects
    dataset_test = Span_Dataset(tok_batch_test, indexes=indexes_test, offsets=offsets_test )
    
    return dataset_test, special_tokens

def postprocess_predictions(preds, dataset, special_tokens, labels=pt_constants.label_dict):
    
    
    predicted_values = torch.Tensor(preds.predictions)
    indexes_test = dataset.indexes
    offsets_test = dataset.offsets
    
    num_labels = len(labels)
    
    cls_ = special_tokens['cls']
    pad = special_tokens['pad']
    
    #provisional
    results_joined = pool_results(predicted_values.tolist(), indexes_test, cls_, pad)

    #calc max for padding
    max_len = max([len(a) for a in results_joined])

    #pad labels -> np.array 
    preds_padded = [list(padded(row, fillvalue=[0]*num_labels, n=max_len)) for row in results_joined]
    results_joined_padded = sigmoid(torch.Tensor(preds_padded))

    to_write_scores = np.array(results_joined_padded)
    
    return to_write_scores


def predict_raw(instance, list_texts):
    """
    Get raw scores from XLM-R, 
    Returns a dictionary {'scores': np.array (documents, tokens, labels )}
    
    Warning: function pads token dimmension to the size of the longest document (can lead to memory issues).
    """

    trainer_for_prediction = instance['trainer']
    tokenizer = instance['tokenizer']
    
    # tokenize text
    dataset_test, special_tokens = tokenize_data_multispan(list_texts, tokenizer)

    # predict
    preds = trainer_for_prediction.predict(dataset_test)

    # post process predictions (how ?)
    scores = postprocess_predictions(preds, dataset_test, special_tokens)

    # save
    
    output = {"scores": scores, "offsets": dataset_test.offsets}
    
#    np.save(OUT_SCORES, to_write_scores)
#    pickle.dump(to_write_offsets, open(OUT_OFFSETS, 'wb'))
    
    return output

def predict(instance, list_texts, **kwargs):
    """
        Main predict function given a threshold
        
        Keyword arguments:
        - cfg: dict from configParser
        - threshold: float, defaults to 0.35, overrides default value in dict

        Returns: dict {'labels': list of np.arrays w/ labels_per_token, 'char_offsets': list of tuples (start,end) w/ offsets per token of 'labels' array}
        all elements of 'labels' and 'char_offsets' are index aligned (both the list and the array elements)
    """
    from pt_postprocessing import postprocess_scores
    
    if 'threshold' in kwargs.keys():
        threshold = kwargs['threshold']
    else:
        threshold = instance['default_threshold']
    
    print('Using threshold: ', threshold)
    
    #get classifier results
    raw_output = predict_raw(instance, list_texts)
    
    ret = postprocess_scores(raw_output, threshold)
    
    return ret

