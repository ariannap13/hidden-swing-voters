from roberta_multispan import *
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback, IntervalStrategy
import torch 
import pandas as pd
import transformers
import numpy as np
import pickle 
import time
from tqdm import tqdm

from more_itertools import chunked, pairwise, padded, windowed, flatten ,unzip

label_dict = dict(pickle.load(open('label_dict-golden.pkl','rb')))
os.environ["WANDB_DISABLED"] = "true"

# HERE set the directory to store the models
MODEL_DIR = './models/'
## HERE Change the paths to the files created with make_df.ipynb

df_train_txt = pd.read_csv('./st3-train_text.csv')
df_train_spans = pd.read_csv('./st3-train_spans.csv')
df_dev_txt = pd.read_csv('./st3-dev_text.csv')
df_dev_spans = pd.read_csv('./st3-dev_spans.csv')
df_test_txt = pd.read_csv('./st3-test_text.csv')
df_test_spans = pd.read_csv('.//st3-test_spans.csv')

## HERE edit the Training Arguments
args = TrainingArguments(
    model_name,
    #evaluation_strategy = "epoch",
    #num_warmup_steps=6000,
    learning_rate=3e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    weight_decay=0.01,
    #num_train_epochs=15,
        
    ## NEW ARGS
    evaluation_strategy = IntervalStrategy.STEPS,
    eval_steps = 150,
    max_steps=4000,
    load_best_model_at_end = True,
    metric_for_best_model = 'f1',
    save_total_limit = 5
        
    )


########### Functions

def compute_metrics_multi(p):
    
    from torch.nn import Sigmoid
    s = Sigmoid()

    #provisional
    predictions, labels = p
    predicted_values = s(torch.Tensor(predictions)) > .5
    labels = np.array(labels)

    correct = (predicted_values & labels).sum()
    prec = correct / (predicted_values.sum() + 0.000001)
    rec = correct / labels.sum()
    f1 = 2*prec*rec/(prec+rec)
    
    pos_mean = predictions[labels == 1].mean()
    neg_mean = predictions[labels == 0].mean()
    
    return {
        "positive_tokens": predicted_values.sum(),
        "pos_mean":pos_mean,
        "neg_mean":neg_mean,
        "difference":pos_mean - neg_mean,
        "precision": prec,
        "recall":rec,
        "f1": 0 if np.isnan(f1) else f1
    }




def make_chunks(docs, cls_, sep, pad):
    
    windows = map(lambda y: windowed(y, n=510, step=255, fillvalue=pad), docs)
    windows = map(lambda y: [[cls_] + list(w) + [sep]  for w in y], windows)
    return list(windows)

def pool_results(predictions, indexes, cls_, pad):
    
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


########### load data 


#LOAD FILES

label_dict = dict(pickle.load(open('label_dict-golden.pkl','rb')))

# HERE add tuple (['lang'], 'model_name') with the langs to keep and the model to train
# - For the ACL model just add ('multi', 'xlm-roberta-large')
# - see commented examples for more

MODEL_NAMES = [('multi', 'xlm-roberta-large')
                #('ru','xlm-roberta-large'),
               #('po','xlm-roberta-large'),
               #(['en','fr','it','ge'], 'xlm-roberta-large'),
               #(['po','ru'], 'xlm-roberta-large'),
            #('po','sdadas/polish-roberta-large-v2'),
               #('po','allegro/herbert-large-cased'),
               #('ru','DeepPavlov/xlm-roberta-large-en-ru'),
               #('ru','sberbank-ai/ruRoberta-large')
              ]

for CURR_LANG, pretrained_model_string in tqdm(MODEL_NAMES):

    if isinstance(CURR_LANG, str):
        curr_train_txt = df_train_txt[df_train_txt.lang == CURR_LANG]
        curr_dev_txt = df_dev_txt[df_dev_txt.lang == CURR_LANG]
        curr_test_txt = df_test_txt[df_test_txt.lang == CURR_LANG]
        curr_train_spans = df_train_spans[df_train_spans.id.isin(curr_train_txt.id)]
        curr_dev_spans = df_dev_spans[df_dev_spans.id.isin(curr_dev_txt.id)]
        curr_test_spans = df_test_spans[df_test_spans.id.isin(curr_test_txt.id)]
    
    elif isinstance(CURR_LANG, list):
    
        curr_train_txt = df_train_txt[df_train_txt.lang.isin(CURR_LANG)]
        curr_dev_txt = df_dev_txt[df_dev_txt.lang.isin(CURR_LANG)]
        curr_test_txt = df_test_txt[df_test_txt.lang.isin(CURR_LANG)]
        curr_train_spans = df_train_spans[df_train_spans.id.isin(curr_train_txt.id)]
        curr_dev_spans = df_dev_spans[df_dev_spans.id.isin(curr_dev_txt.id)]
        curr_test_spans = df_test_spans[df_test_spans.id.isin(curr_test_txt.id)]
        
        CURR_LANG = '_'.join(CURR_LANG) # fix later errors (use in fname)
        
    
    if CURR_LANG == 'multi':
        curr_train_txt = df_train_txt
        curr_dev_txt = df_dev_txt
        curr_test_txt = df_test_txt
        curr_train_spans = df_train_spans
        curr_dev_spans = df_dev_spans
        curr_test_spans = df_test_spans
               
    print('Total no. of articles: \t',curr_train_txt.shape[0])
    train_txt, train_spans, indexes_train = preprocess_files(curr_train_txt, curr_train_spans, './label_dict-golden.pkl')
    dev_txt, dev_spans, indexes_dev = preprocess_files(curr_dev_txt, curr_dev_spans, './label_dict-golden.pkl')
    test_txt, test_spans, indexes_test = preprocess_files(curr_test_txt, curr_test_spans, './label_dict-golden.pkl')
    
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_string)

    #Train Data
    tok_batch = tokenizer(train_txt,padding=False, truncation=False, add_special_tokens=False)
    offsets= list(map(lambda x: x.offsets, tok_batch.encodings))
    train_y = list(map(calculate_y, zip(offsets,train_spans)))
    train_y = [y + [([0] * 23)] for y in train_y] #add label for SEP token
    input_ids_train = [ids + [2] for ids in tok_batch['input_ids']]

    #tok_batch['labels'] = train_y

    #Dev Data

    tok_batch_dev = tokenizer(dev_txt,padding=False, truncation=False, add_special_tokens=False)
    offsets_dev= list(map(lambda x: x.offsets, tok_batch_dev.encodings))
    dev_y = list(map(calculate_y, zip(offsets_dev, dev_spans)))
    dev_y = [y + [([0] * 23)] for y in dev_y] #add label for SEP token
    #dev_y=[[[tok[11],tok[12]] for tok in y] for y in dev_y] #only LL
    #tok_batch_dev['labels'] = dev_y
    input_ids_dev = [ids + [2] for ids in tok_batch_dev['input_ids']]

    #Test Data

    tok_batch_test = tokenizer(test_txt,padding=False, truncation=False, add_special_tokens=False)
    offsets_test= list(map(lambda x: x.offsets, tok_batch_test.encodings))
    test_y = list(map(calculate_y, zip(offsets_test, test_spans)))
    test_y = [y + [([0] * 23)] for y in test_y] #add label for SEP token
    #dev_y=[[[tok[11],tok[12]] for tok in y] for y in dev_y] #only LL
    #tok_batch_dev['labels'] = dev_y
    input_ids_test = [ids + [2] for ids in tok_batch_test['input_ids']]
    
    
    cls_ = [0] * 23
    sep = [0] * 23
    pad = [0] * 23

    ### CHUNK

    #Chunk Train
    ## IndexIDS
    cls_ = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    windows = make_chunks(input_ids_train, cls_, sep, pad)
    indexes = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    index_ids_chunked = windows_flat
    ## labels
    cls_ = [0] * 23
    sep = [0] * 23
    pad = [0] * 23

    windows = make_chunks(train_y, cls_, sep, pad)
    indexes = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    labels_chunked = windows_flat

    ## Att masks
    att_mask_chunked = [[1] * len(chu) for chu in index_ids_chunked]


    ## put back on the Tokenizer Batch
    tok_batch['input_ids'] = index_ids_chunked
    tok_batch['attention_mask'] = att_mask_chunked
    tok_batch['labels'] = labels_chunked

    #Chunk DEV
    ## IndexIDS
    cls_ = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    windows = make_chunks(input_ids_dev, cls_, sep, pad)
    indexes_dev = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    index_ids_chunked_dev = windows_flat
    ## labels
    cls_ = [0] * 23
    sep = [0] * 23
    pad = [0] * 23

    windows = make_chunks(dev_y, cls_, sep, pad)
    indexes_dev = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    labels_chunked_dev = windows_flat

    ## Att masks
    att_mask_chunked_dev = [[1] * len(chu) for chu in index_ids_chunked_dev]


    ## put back on the Tokenizer Batch
    tok_batch_dev['input_ids'] = index_ids_chunked_dev
    tok_batch_dev['attention_mask'] = att_mask_chunked_dev
    tok_batch_dev['labels'] = labels_chunked_dev

    #Chunk TEst
    ## IndexIDS
    cls_ = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    windows = make_chunks(input_ids_test, cls_, sep, pad)
    indexes_test = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    index_ids_chunked_test = windows_flat
    

    ## labels
    cls_ = [0] * 23
    sep = [0] * 23
    pad = [0] * 23

    windows = make_chunks(test_y, cls_, sep, pad)
    indexes_test = list(flatten([[i]*len(w) for i,w in  enumerate(windows)]))
    windows_flat = list(flatten(windows))
    labels_chunked_test = windows_flat
    
    

    ## Att masks
    att_mask_chunked_test = [[1] * len(chu) for chu in index_ids_chunked_test]
    
    #dump indexes for joining
    #pickle.dump(indexes_train, open(CURR_LANG + '_ind_train.pkl', 'wb'))
    #pickle.dump(indexes_dev, open(CURR_LANG + '_ind_dev.pkl', 'wb'))
    #pickle.dump(indexes_test, open(CURR_LANG + '_ind_test.pkl', 'wb'))
    
    
    
    ## put back on the Tokenizer Batch
    tok_batch_test['input_ids'] = index_ids_chunked_test
    tok_batch_test['attention_mask'] = att_mask_chunked_test
    tok_batch_test['labels'] = labels_chunked_test
    
    #Final objects
    dataset_train = Span_Dataset(tok_batch)
    dataset_dev = Span_Dataset(tok_batch_dev)
    dataset_test = Span_Dataset(tok_batch_test)
    
    ###### TRAIN MODEL
    
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model_string, num_labels=len(label_dict))
    

    torch.cuda.empty_cache()
    device = torch.device("cuda")

    model.to(device)

    now = '_'.join([str(i) for i in [time.gmtime().tm_mday,time.gmtime().tm_hour,time.gmtime().tm_min]])
    model_name = MODEL_DIR + pretrained_model_string.replace('/','_') + '_lr3_' + CURR_LANG + now

    #Test Training

    trainer = MultiLabelTrainer(
        model,
        args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_multi,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
      
    )

    trainer.train()
    trainer.save_model()
    preds = trainer.predict(dataset_test)
    # Save the results of the test dataset (logits)
    torch.save(preds, CURR_LANG + '_' + now + '.pt')
    