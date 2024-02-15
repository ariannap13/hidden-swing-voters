from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import torch
import os
import pandas as pd
import pickle 

os.environ["WANDB_DISABLED"] = "true"

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
        
    return (to_tokenize, to_make_mask, indexes) # (list of strings, list of dicts {label:[list of tuples]})

#Definitions 
## TODO: Refactor in .py --> Trainer, DataCollator, Mode subclass
class MultiLabelTrainer(Trainer):
    #Adapted from https://discuss.huggingface.co/t/multi-label-token-classification/16509/8
    def __init__(self, *args, class_weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            logging.info(f"Using multi-label classification with class weights", class_weights)
        self.loss_fct = BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        
        labels = labels.float() #so torch does not compaing about casting
        try:
            loss = self.loss_fct(outputs.logits.view(-1), labels.view(-1))
        except AttributeError:  # DataParallel
            loss = self.loss_fct(outputs.module.logits.view(-1), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
    
#Preliminary --> make it with map/numpy if slow
def calculate_y(inp, max_length = None, no_of_labels = 23):
    
    offsets, span_dict = inp
    #mask_y = torch.zeros((max_length,no_of_labels), dtype=torch.int32) 
    
    if not max_length: 
        max_length = len(offsets)
        
    mask_y = [[0] * no_of_labels for i in  range(max_length)]
    for label,spans in span_dict.items():
        for (span_start, span_end) in spans:
            iterator = filter(lambda offset:  (span_start <= offset[1][0]) & (offset[1][1] <= span_end), enumerate(offsets))
            #ys = [subtoken for subtoken, chars in iterator]
            #mask_y[:,label].put_(ys, 1.0) #numpy
            #mask_y[ys,label] = 1
            for subtoken, chars in iterator:
                mask_y[subtoken][label] = 1
    
    return mask_y

def infer_negatives(mask_y):
    """
    Infer the negative class (No Persuation Technique) in tokens without labels
    """
    other_label = torch.zeros((mask_y.shape[0],1))
    other_label[mask_y.sum(axis=1) == 0.0, 0] = 1.0
    mask_y = torch.cat([mask_y, other_label], axis=1)
    
    return mask_y



class Span_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, indexes=None, offsets=None):
        self.encodings = encodings
        self.input_ids = self.encodings['input_ids']
        self.labels = self.encodings['labels'] if 'labels' in encodings else None
        self.attention_mask = self.encodings['attention_mask']
        self.indexes = indexes if indexes else range(len(encodings)) #which original document this chunk corresponds
        self.offsets = offsets 
        
    def __getitem__(self, idx):
        
        input_ids = list(self.input_ids[idx])
        
        attention_mask = list(self.attention_mask[idx])
        
        ret = {'input_ids': input_ids, 
               'attention_mask':attention_mask}
        
        if self.labels:
            ret['labels'] = list(self.labels[idx]),
        return ret

    def __len__(self):
        return len(self.encodings.input_ids)
